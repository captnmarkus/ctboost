package io.github.ctboost.inference;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Locale;
import java.util.Objects;

/**
 * Immutable, thread-safe scorer for prepared-feature CTBoost JSON predictors.
 *
 * <p>The loader validates dimensions, indices, and tree topology before any
 * prediction. Raw feature-pipeline execution is intentionally unsupported.</p>
 */
public final class JsonPredictor {
    private static final String ARTIFACT_FORMAT = "ctboost-json-predictor";
    private static final long DEFAULT_MAX_ARTIFACT_BYTES = 512L * 1024L * 1024L;

    private static final ObjectMapper MAPPER = createMapper();

    private final String objectiveName;
    private final double learningRate;
    private final double[] treeLearningRates;
    private final double[] baseScore;
    private final int predictionDimension;
    private final int numFeatures;
    private final Quantization quantization;
    private final Tree[] trees;
    private final JsonNode classLabels;
    private final JsonNode inferenceManifest;

    private JsonPredictor(ObjectNode root) {
        String format = requiredText(root, "format", "predictor");
        if (!ARTIFACT_FORMAT.equals(format)) {
            throw invalid("unsupported predictor format: " + format);
        }
        int formatVersion = requiredInt(root, "format_version", "predictor");
        if (formatVersion != 1 && formatVersion != 2) {
            throw invalid("unsupported predictor format version: " + formatVersion);
        }
        boolean prepared = requiredBoolean(root, "expects_prepared_features", "predictor");
        if (!prepared) {
            throw invalid(
                    "JVM inference supports prepared numeric features only; "
                            + "raw feature_pipeline_state execution is not supported");
        }

        this.objectiveName = requiredText(root, "objective_name", "predictor");
        if (objectiveName.isEmpty()) {
            throw invalid("objective_name must not be empty");
        }
        this.learningRate = requiredFiniteDouble(root, "learning_rate", "predictor");
        this.predictionDimension = requiredInt(root, "prediction_dimension", "predictor");
        if (predictionDimension <= 0) {
            throw invalid("prediction_dimension must be positive");
        }
        String normalizedObjective = normalizeObjective(objectiveName);
        if (isBinaryObjective(normalizedObjective) && predictionDimension != 1) {
            throw invalid("binary objectives require prediction_dimension == 1");
        }
        if (isMulticlassObjective(normalizedObjective) && predictionDimension < 2) {
            throw invalid("multiclass objectives require prediction_dimension >= 2");
        }
        this.numFeatures = requiredInt(root, "num_features", "predictor");
        if (numFeatures < 0) {
            throw invalid("num_features must be non-negative");
        }
        this.baseScore = finiteDoubleArray(
                requiredArray(root, "base_score", "predictor"), "base_score");
        if (baseScore.length != predictionDimension) {
            throw invalid("base_score length must match prediction_dimension");
        }
        JsonNode ratesNode = root.get("tree_learning_rates");
        this.treeLearningRates = ratesNode == null || ratesNode.isNull()
                ? new double[0]
                : finiteDoubleArray(requireArrayNode(ratesNode, "tree_learning_rates"),
                        "tree_learning_rates");

        this.quantization = Quantization.parse(
                requiredObject(root, "quantization_schema", "predictor"), numFeatures);
        ArrayNode treeNodes = requiredArray(root, "trees", "predictor");
        if (treeNodes.isEmpty()) {
            throw invalid("predictor must contain at least one tree");
        }
        if (treeNodes.size() % predictionDimension != 0) {
            throw invalid("tree count must be divisible by prediction_dimension");
        }
        int iterationCount = treeNodes.size() / predictionDimension;
        if (treeLearningRates.length > iterationCount) {
            throw invalid("tree_learning_rates cannot exceed the iteration count");
        }
        this.trees = new Tree[treeNodes.size()];
        for (int index = 0; index < trees.length; ++index) {
            JsonNode treeNode = treeNodes.get(index);
            if (!(treeNode instanceof ObjectNode treeObject)) {
                throw invalid("trees[" + index + "] must be an object");
            }
            trees[index] = Tree.parse(treeObject, quantization, index);
        }

        JsonNode labels = root.get("class_labels");
        this.classLabels = labels == null || labels.isNull() ? null : labels.deepCopy();
        validateClassLabels();
        JsonNode manifest = root.get("inference_manifest");
        if (manifest != null && !manifest.isNull() && !manifest.isObject()) {
            throw invalid("inference_manifest must be an object or null");
        }
        this.inferenceManifest = manifest == null || manifest.isNull()
                ? null
                : manifest.deepCopy();
    }

    /** Load a predictor with a 512 MiB artifact-size limit. */
    public static JsonPredictor load(Path path) throws IOException {
        return load(path, DEFAULT_MAX_ARTIFACT_BYTES);
    }

    /** Load a predictor with an explicit positive artifact-size limit. */
    public static JsonPredictor load(Path path, long maxArtifactBytes) throws IOException {
        Objects.requireNonNull(path, "path");
        if (maxArtifactBytes <= 0L) {
            throw new IllegalArgumentException("maxArtifactBytes must be positive");
        }
        JsonNode document;
        try (var stream = new BoundedInputStream(Files.newInputStream(path), maxArtifactBytes)) {
            document = MAPPER.readTree(stream);
        } catch (ArtifactTooLargeException exception) {
            throw invalid("predictor artifact exceeds the configured size limit");
        } catch (IOException exception) {
            throw new InvalidModelException("could not parse predictor JSON", exception);
        }
        if (!(document instanceof ObjectNode root)) {
            throw invalid("predictor document must be a JSON object");
        }
        return new JsonPredictor(root);
    }

    public int numFeatures() {
        return numFeatures;
    }

    public int predictionDimension() {
        return predictionDimension;
    }

    public String objectiveName() {
        return objectiveName;
    }

    /** Return a defensive copy of the embedded inference manifest, or null. */
    public JsonNode inferenceManifest() {
        return inferenceManifest == null ? null : inferenceManifest.deepCopy();
    }

    /** Score one prepared numeric row and return one raw margin per output. */
    public double[] predictRaw(float[] row) {
        Objects.requireNonNull(row, "row");
        if (row.length != numFeatures) {
            throw new IllegalArgumentException(
                    "expected " + numFeatures + " features, got " + row.length);
        }
        int[] bins = quantization.binRow(row);
        double[] scores = Arrays.copyOf(baseScore, baseScore.length);
        for (int treeIndex = 0; treeIndex < trees.length; ++treeIndex) {
            int iterationIndex = treeIndex / predictionDimension;
            double scale = iterationIndex < treeLearningRates.length
                    ? treeLearningRates[iterationIndex]
                    : learningRate;
            scores[treeIndex % predictionDimension] += scale * trees[treeIndex].score(bins);
        }
        return scores;
    }

    /** Score a batch of prepared numeric rows. */
    public double[][] predictRaw(float[][] rows) {
        Objects.requireNonNull(rows, "rows");
        double[][] result = new double[rows.length][];
        for (int index = 0; index < rows.length; ++index) {
            result[index] = predictRaw(rows[index]);
        }
        return result;
    }

    /** Convenience scalar raw prediction for one-dimensional models. */
    public double predictRawScalar(float[] row) {
        if (predictionDimension != 1) {
            throw new IllegalStateException("predictRawScalar requires prediction_dimension == 1");
        }
        return predictRaw(row)[0];
    }

    /** Return binary or multiclass probabilities for one row. */
    public double[] predictProba(float[] row) {
        double[] raw = predictRaw(row);
        String normalized = normalizeObjective(objectiveName);
        if (isBinaryObjective(normalized)) {
            double positive = sigmoid(raw[0]);
            return new double[] {1.0 - positive, positive};
        }
        if (isMulticlassObjective(normalized)) {
            double maximum = raw[0];
            for (int index = 1; index < raw.length; ++index) {
                maximum = Math.max(maximum, raw[index]);
            }
            double total = 0.0;
            double[] probabilities = new double[raw.length];
            for (int index = 0; index < raw.length; ++index) {
                probabilities[index] = Math.exp(raw[index] - maximum);
                total += probabilities[index];
            }
            for (int index = 0; index < probabilities.length; ++index) {
                probabilities[index] /= total;
            }
            return probabilities;
        }
        throw new IllegalStateException(
                "predictProba is only available for classification objectives");
    }

    /** Return the zero-based maximum-probability class index. */
    public int predictClassIndex(float[] row) {
        double[] probabilities = predictProba(row);
        int best = 0;
        for (int index = 1; index < probabilities.length; ++index) {
            if (probabilities[index] > probabilities[best]) {
                best = index;
            }
        }
        return best;
    }

    /** Return the embedded class label, or a JSON integer index when labels are absent. */
    public JsonNode predictClassLabel(float[] row) {
        int index = predictClassIndex(row);
        if (classLabels == null) {
            return MAPPER.getNodeFactory().numberNode(index);
        }
        return classLabels.get(index).deepCopy();
    }

    private void validateClassLabels() {
        if (classLabels == null) {
            return;
        }
        if (!classLabels.isArray()) {
            throw invalid("class_labels must be an array or null");
        }
        String objective = normalizeObjective(objectiveName);
        int expected;
        if (isBinaryObjective(objective)) {
            expected = 2;
        } else if (isMulticlassObjective(objective)) {
            expected = predictionDimension;
        } else {
            throw invalid("class_labels are only valid for classification objectives");
        }
        if (classLabels.size() != expected) {
            throw invalid("class_labels length does not match the probability dimension");
        }
    }

    private static ObjectMapper createMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION);
        mapper.enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
        return mapper;
    }

    private static final class ArtifactTooLargeException extends IOException {
        private ArtifactTooLargeException() {
            super("artifact byte limit exceeded");
        }
    }

    private static final class BoundedInputStream extends FilterInputStream {
        private final long maximumBytes;
        private long bytesRead;

        private BoundedInputStream(InputStream input, long maximumBytes) {
            super(input);
            this.maximumBytes = maximumBytes;
        }

        private void account(long count) throws ArtifactTooLargeException {
            if (count <= 0L) {
                return;
            }
            if (bytesRead > maximumBytes - count) {
                throw new ArtifactTooLargeException();
            }
            bytesRead += count;
        }

        @Override
        public int read() throws IOException {
            int value = super.read();
            if (value >= 0) {
                account(1L);
            }
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            if (length == 0) {
                return 0;
            }
            long remaining = maximumBytes - bytesRead;
            int boundedLength = remaining >= length ? length : (int) (remaining + 1L);
            int count = super.read(buffer, offset, boundedLength);
            account(count);
            return count;
        }

        @Override
        public long skip(long count) throws IOException {
            if (count <= 0L) {
                return 0L;
            }
            long remaining = maximumBytes - bytesRead;
            long boundedCount = remaining >= count ? count : remaining + 1L;
            long skipped = super.skip(boundedCount);
            account(skipped);
            return skipped;
        }

        @Override
        public boolean markSupported() {
            return false;
        }
    }

    private static String normalizeObjective(String value) {
        return value.trim().toLowerCase(Locale.ROOT);
    }

    private static boolean isBinaryObjective(String value) {
        return value.equals("logloss")
                || value.equals("binary_logloss")
                || value.equals("binary:logistic");
    }

    private static boolean isMulticlassObjective(String value) {
        return value.equals("multiclass")
                || value.equals("softmax")
                || value.equals("softmaxloss");
    }

    private static double sigmoid(double value) {
        if (value >= 0.0) {
            double exponential = Math.exp(-value);
            return 1.0 / (1.0 + exponential);
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }

    private static InvalidModelException invalid(String message) {
        return new InvalidModelException(message);
    }

    private static JsonNode required(ObjectNode object, String field, String context) {
        JsonNode value = object.get(field);
        if (value == null || value.isNull()) {
            throw invalid(context + " is missing " + field);
        }
        return value;
    }

    private static ObjectNode requiredObject(ObjectNode object, String field, String context) {
        JsonNode value = required(object, field, context);
        if (!(value instanceof ObjectNode result)) {
            throw invalid(context + "." + field + " must be an object");
        }
        return result;
    }

    private static ArrayNode requiredArray(ObjectNode object, String field, String context) {
        return requireArrayNode(required(object, field, context), context + "." + field);
    }

    private static ArrayNode requireArrayNode(JsonNode value, String context) {
        if (!(value instanceof ArrayNode result)) {
            throw invalid(context + " must be an array");
        }
        return result;
    }

    private static String requiredText(ObjectNode object, String field, String context) {
        JsonNode value = required(object, field, context);
        if (!value.isTextual()) {
            throw invalid(context + "." + field + " must be a string");
        }
        return value.textValue();
    }

    private static boolean requiredBoolean(ObjectNode object, String field, String context) {
        JsonNode value = required(object, field, context);
        if (!value.isBoolean()) {
            throw invalid(context + "." + field + " must be a boolean");
        }
        return value.booleanValue();
    }

    private static int requiredInt(ObjectNode object, String field, String context) {
        JsonNode value = required(object, field, context);
        Integer result = exactIntOrNull(value);
        if (result == null) {
            throw invalid(context + "." + field + " must be a 32-bit integer");
        }
        return result;
    }

    private static double requiredFiniteDouble(ObjectNode object, String field, String context) {
        JsonNode value = required(object, field, context);
        if (!value.isNumber()) {
            throw invalid(context + "." + field + " must be numeric");
        }
        double result = value.doubleValue();
        if (!Double.isFinite(result)) {
            throw invalid(context + "." + field + " must be finite");
        }
        return result;
    }

    private static int[] intArray(ArrayNode values, String context) {
        int[] result = new int[values.size()];
        for (int index = 0; index < result.length; ++index) {
            JsonNode value = values.get(index);
            Integer parsed = exactIntOrNull(value);
            if (parsed == null) {
                throw invalid(context + "[" + index + "] must be a 32-bit integer");
            }
            result[index] = parsed;
        }
        return result;
    }

    private static Integer exactIntOrNull(JsonNode value) {
        if (!value.isNumber() || !Double.isFinite(value.doubleValue())) {
            return null;
        }
        try {
            return value.decimalValue().intValueExact();
        } catch (ArithmeticException exception) {
            return null;
        }
    }

    private static int[] bitArray(ArrayNode values, String context) {
        int[] result = new int[values.size()];
        for (int index = 0; index < result.length; ++index) {
            JsonNode value = values.get(index);
            if (value.isBoolean()) {
                result[index] = value.booleanValue() ? 1 : 0;
            } else {
                Integer parsed = exactIntOrNull(value);
                if (parsed == null || (parsed != 0 && parsed != 1)) {
                    throw invalid(context + "[" + index + "] must be 0, 1, false, or true");
                }
                result[index] = parsed;
            }
        }
        return result;
    }

    private static double[] finiteDoubleArray(ArrayNode values, String context) {
        double[] result = new double[values.size()];
        for (int index = 0; index < result.length; ++index) {
            JsonNode value = values.get(index);
            if (!value.isNumber()) {
                throw invalid(context + "[" + index + "] must be numeric");
            }
            result[index] = value.doubleValue();
            if (!Double.isFinite(result[index])) {
                throw invalid(context + "[" + index + "] must be finite");
            }
        }
        return result;
    }

    private static final class Quantization {
        private final int[] bins;
        private final int[] cutOffsets;
        private final double[] cutValues;
        private final int[] categorical;
        private final int[] missing;
        private final int defaultNanMode;
        private final int[] nanModes;

        private Quantization(
                int[] bins,
                int[] cutOffsets,
                double[] cutValues,
                int[] categorical,
                int[] missing,
                int defaultNanMode,
                int[] nanModes) {
            this.bins = bins;
            this.cutOffsets = cutOffsets;
            this.cutValues = cutValues;
            this.categorical = categorical;
            this.missing = missing;
            this.defaultNanMode = defaultNanMode;
            this.nanModes = nanModes;
        }

        private static Quantization parse(ObjectNode object, int numFeatures) {
            int[] bins = intArray(
                    requiredArray(object, "num_bins_per_feature", "quantization_schema"),
                    "quantization_schema.num_bins_per_feature");
            int[] offsets = intArray(
                    requiredArray(object, "cut_offsets", "quantization_schema"),
                    "quantization_schema.cut_offsets");
            double[] cuts = finiteDoubleArray(
                    requiredArray(object, "cut_values", "quantization_schema"),
                    "quantization_schema.cut_values");
            int[] categorical = bitArray(
                    requiredArray(object, "categorical_mask", "quantization_schema"),
                    "quantization_schema.categorical_mask");
            int[] missing = bitArray(
                    requiredArray(object, "missing_value_mask", "quantization_schema"),
                    "quantization_schema.missing_value_mask");
            int nanMode = requiredInt(object, "nan_mode", "quantization_schema");
            if (nanMode < 0 || nanMode > 2) {
                throw invalid("quantization_schema.nan_mode must be 0, 1, or 2");
            }
            JsonNode nanModesNode = object.get("nan_modes");
            int[] nanModes = nanModesNode == null || nanModesNode.isNull()
                    ? new int[0]
                    : intArray(requireArrayNode(nanModesNode, "quantization_schema.nan_modes"),
                            "quantization_schema.nan_modes");

            if (bins.length != numFeatures
                    || categorical.length != numFeatures
                    || missing.length != numFeatures) {
                throw invalid("quantization feature arrays must match num_features");
            }
            if (offsets.length != numFeatures + 1) {
                throw invalid("cut_offsets length must be num_features + 1");
            }
            if (nanModes.length != 0 && nanModes.length != numFeatures) {
                throw invalid("nan_modes must be empty or match num_features");
            }
            if (offsets[0] != 0 || offsets[numFeatures] != cuts.length) {
                throw invalid("cut_offsets must start at zero and end at cut_values length");
            }
            for (int feature = 0; feature < numFeatures; ++feature) {
                if (bins[feature] < 0 || bins[feature] > 65535) {
                    throw invalid("num_bins_per_feature is outside uint16 range");
                }
                if (offsets[feature] < 0 || offsets[feature] > offsets[feature + 1]
                        || offsets[feature + 1] > cuts.length) {
                    throw invalid("cut_offsets must be monotone and in range");
                }
                int featureNanMode = nanModes.length == 0 ? nanMode : nanModes[feature];
                if (featureNanMode < 0 || featureNanMode > 2) {
                    throw invalid("nan_modes entries must be 0, 1, or 2");
                }
                int nonMissingBins = bins[feature] - missing[feature];
                if (nonMissingBins < 0) {
                    throw invalid("missing-value bin count exceeds total bins");
                }
                int cutCount = offsets[feature + 1] - offsets[feature];
                int expectedCuts = categorical[feature] != 0
                        ? nonMissingBins
                        : Math.max(nonMissingBins - 1, 0);
                if (cutCount != expectedCuts) {
                    throw invalid("cut count is inconsistent with feature bin metadata");
                }
                for (int cut = offsets[feature] + 1; cut < offsets[feature + 1]; ++cut) {
                    if (!(cuts[cut] > cuts[cut - 1])) {
                        throw invalid("feature cuts must be strictly increasing");
                    }
                }
            }
            return new Quantization(
                    bins, offsets, cuts, categorical, missing, nanMode, nanModes);
        }

        private int[] binRow(float[] row) {
            int[] result = new int[bins.length];
            for (int feature = 0; feature < bins.length; ++feature) {
                result[feature] = binValue(feature, row[feature]);
            }
            return result;
        }

        private int binValue(int feature, float value) {
            int binCount = bins[feature];
            if (binCount == 0) {
                return 0;
            }
            int nanMode = nanModes.length == 0 ? defaultNanMode : nanModes[feature];
            if (Float.isNaN(value)) {
                return nanMode == 2 ? binCount - 1 : 0;
            }
            int nonMissingBins = binCount - missing[feature];
            if (nonMissingBins == 0) {
                return nanMode == 2 ? binCount - 1 : 0;
            }
            int offset = missing[feature] != 0 && nanMode == 1 ? 1 : 0;
            int begin = cutOffsets[feature];
            int end = cutOffsets[feature + 1];
            if (categorical[feature] != 0) {
                int insertion = lowerBound(cutValues, begin, end, value) - begin;
                return offset + Math.min(insertion, nonMissingBins - 1);
            }
            return offset + upperBound(cutValues, begin, end, value) - begin;
        }

        private static int lowerBound(double[] values, int begin, int end, float target) {
            int left = begin;
            int right = end;
            while (left < right) {
                int middle = left + (right - left) / 2;
                if (values[middle] < target) {
                    left = middle + 1;
                } else {
                    right = middle;
                }
            }
            return left;
        }

        private static int upperBound(double[] values, int begin, int end, float target) {
            int left = begin;
            int right = end;
            while (left < right) {
                int middle = left + (right - left) / 2;
                if (target < values[middle]) {
                    right = middle;
                } else {
                    left = middle + 1;
                }
            }
            return left;
        }
    }

    private static final class Node {
        private final boolean leaf;
        private final boolean categorical;
        private final int feature;
        private final int splitBin;
        private final int left;
        private final int right;
        private final double weight;
        private final int[] leftCategories;

        private Node(
                boolean leaf,
                boolean categorical,
                int feature,
                int splitBin,
                int left,
                int right,
                double weight,
                int[] leftCategories) {
            this.leaf = leaf;
            this.categorical = categorical;
            this.feature = feature;
            this.splitBin = splitBin;
            this.left = left;
            this.right = right;
            this.weight = weight;
            this.leftCategories = leftCategories;
        }
    }

    private static final class Tree {
        private final Node[] nodes;

        private Tree(Node[] nodes) {
            this.nodes = nodes;
        }

        private static Tree parse(ObjectNode tree, Quantization quantization, int treeIndex) {
            ArrayNode values = requiredArray(tree, "nodes", "trees[" + treeIndex + "]");
            if (values.isEmpty()) {
                throw invalid("trees[" + treeIndex + "] must contain nodes");
            }
            Node[] nodes = new Node[values.size()];
            for (int index = 0; index < nodes.length; ++index) {
                JsonNode value = values.get(index);
                if (!(value instanceof ObjectNode object)) {
                    throw invalid("tree node must be an object");
                }
                String context = "trees[" + treeIndex + "].nodes[" + index + "]";
                boolean leaf = requiredBoolean(object, "is_leaf", context);
                boolean categorical = requiredBoolean(object, "is_categorical_split", context);
                int feature = requiredInt(object, "split_feature_id", context);
                int splitBin = requiredInt(object, "split_bin_index", context);
                int left = requiredInt(object, "left_child", context);
                int right = requiredInt(object, "right_child", context);
                double weight = requiredFiniteDouble(object, "leaf_weight", context);
                int[] routes = bitArray(requiredArray(object, "left_categories", context),
                        context + ".left_categories");

                if (leaf) {
                    if (left != -1 || right != -1) {
                        throw invalid(context + " leaf children must be -1");
                    }
                } else {
                    if (feature < 0 || feature >= quantization.bins.length) {
                        throw invalid(context + " split feature is out of range");
                    }
                    if (left < 0 || left >= nodes.length || right < 0 || right >= nodes.length
                            || left == right) {
                        throw invalid(context + " child index is invalid");
                    }
                    if (splitBin < 0 || splitBin >= quantization.bins[feature]) {
                        throw invalid(context + " split bin is out of range");
                    }
                    if (categorical) {
                        if (quantization.categorical[feature] == 0) {
                            throw invalid(context + " categorical split uses a numeric feature");
                        }
                        if (routes.length < quantization.bins[feature]) {
                            throw invalid(context + " categorical routes do not cover all bins");
                        }
                    } else if (quantization.categorical[feature] != 0) {
                        throw invalid(context + " numeric split uses a categorical feature");
                    }
                }
                nodes[index] = new Node(
                        leaf, categorical, feature, splitBin, left, right, weight, routes);
            }
            validateTopology(nodes, treeIndex);
            return new Tree(nodes);
        }

        private static void validateTopology(Node[] nodes, int treeIndex) {
            boolean[] visited = new boolean[nodes.length];
            Deque<Integer> pending = new ArrayDeque<>();
            pending.push(0);
            int count = 0;
            while (!pending.isEmpty()) {
                int index = pending.pop();
                if (visited[index]) {
                    throw invalid("trees[" + treeIndex + "] contains a cycle or shared child");
                }
                visited[index] = true;
                ++count;
                Node node = nodes[index];
                if (!node.leaf) {
                    pending.push(node.right);
                    pending.push(node.left);
                }
            }
            if (count != nodes.length) {
                throw invalid("trees[" + treeIndex + "] contains unreachable nodes");
            }
        }

        private double score(int[] bins) {
            int nodeIndex = 0;
            for (int steps = 0; steps < nodes.length; ++steps) {
                Node node = nodes[nodeIndex];
                if (node.leaf) {
                    return node.weight;
                }
                int bin = bins[node.feature];
                boolean goLeft = node.categorical
                        ? node.leftCategories[bin] != 0
                        : bin <= node.splitBin;
                nodeIndex = goLeft ? node.left : node.right;
            }
            throw new IllegalStateException("validated tree traversal exceeded its node count");
        }
    }
}
