package io.github.ctboost.inference;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class JsonPredictorConformanceTest {
    private static final double TOLERANCE = 1.0e-12;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void regressionV1MatchesGoldenRowsIncludingNaNAndCategoricalRouting() throws Exception {
        JsonNode cases = read("prepared_regression_v1.cases.json");
        JsonPredictor predictor = JsonPredictor.load(
                fixture(cases.get("artifact").textValue()));
        float[][] rows = rows(cases.get("rows"));
        JsonNode expected = cases.get("raw_predictions");

        assertEquals(2, predictor.numFeatures());
        assertEquals(1, predictor.predictionDimension());
        for (int index = 0; index < rows.length; ++index) {
            assertEquals(
                    expected.get(index).doubleValue(),
                    predictor.predictRawScalar(rows[index]),
                    TOLERANCE);
        }
    }

    @Test
    void binaryV2MatchesRawProbabilityClassAndPerIterationRates() throws Exception {
        JsonNode cases = read("prepared_binary_v2.cases.json");
        JsonPredictor predictor = JsonPredictor.load(
                fixture(cases.get("artifact").textValue()));
        float[][] rows = rows(cases.get("rows"));

        for (int index = 0; index < rows.length; ++index) {
            assertEquals(
                    cases.get("raw_predictions").get(index).doubleValue(),
                    predictor.predictRawScalar(rows[index]),
                    TOLERANCE);
            assertArrayEquals(
                    doubles(cases.get("probabilities").get(index)),
                    predictor.predictProba(rows[index]),
                    TOLERANCE);
            assertEquals(
                    cases.get("class_indices").get(index).intValue(),
                    predictor.predictClassIndex(rows[index]));
            assertEquals(
                    cases.get("class_labels").get(index),
                    predictor.predictClassLabel(rows[index]));
        }
    }

    @Test
    void multiclassV2MatchesRawSoftmaxAndLabels() throws Exception {
        JsonNode cases = read("prepared_multiclass_v2.cases.json");
        JsonPredictor predictor = JsonPredictor.load(
                fixture(cases.get("artifact").textValue()));
        float[][] rows = rows(cases.get("rows"));

        for (int index = 0; index < rows.length; ++index) {
            assertArrayEquals(
                    doubles(cases.get("raw_predictions").get(index)),
                    predictor.predictRaw(rows[index]),
                    TOLERANCE);
            assertArrayEquals(
                    doubles(cases.get("probabilities").get(index)),
                    predictor.predictProba(rows[index]),
                    TOLERANCE);
            assertEquals(
                    cases.get("class_indices").get(index).intValue(),
                    predictor.predictClassIndex(rows[index]));
            assertEquals(
                    cases.get("class_labels").get(index),
                    predictor.predictClassLabel(rows[index]));
        }
    }

    @Test
    void rawPipelineArtifactsFailClosed() {
        InvalidModelException error = assertThrows(
                InvalidModelException.class,
                () -> JsonPredictor.load(fixture("raw_pipeline_v2.json")));
        assertEquals(true, error.getMessage().contains("prepared numeric features only"));
    }

    @Test
    void vectorPredictorFormat3IsRejectedBeforeScoring() throws Exception {
        ObjectNode document = (ObjectNode) read("prepared_multiclass_v2.json");
        document.put("format_version", 3);
        document.put("multi_strategy", "multi_output_tree");
        Path artifact = Files.createTempFile("ctboost-vector-version", ".json");
        try {
            MAPPER.writeValue(artifact.toFile(), document);
            InvalidModelException error = assertThrows(
                    InvalidModelException.class, () -> JsonPredictor.load(artifact));
            assertTrue(error.getMessage().contains("unsupported predictor format version: 3"));
        } finally {
            Files.deleteIfExists(artifact);
        }
    }

    @Test
    void duplicateKeysAreRejected() {
        assertThrows(
                InvalidModelException.class,
                () -> JsonPredictor.load(fixture("duplicate_prepared_flag_v2.json")));
    }

    @Test
    void artifactLimitIsEnforcedAtTheExactByteBoundary() throws Exception {
        Path artifact = fixture("prepared_binary_v2.json");
        long exactSize = Files.size(artifact);
        assertDoesNotThrow(() -> JsonPredictor.load(artifact, exactSize));
        InvalidModelException error = assertThrows(
                InvalidModelException.class,
                () -> JsonPredictor.load(artifact, exactSize - 1L));
        assertTrue(error.getMessage().contains("size limit"));
    }

    @Test
    void schemaIntegerNumbersAcceptExactDecimalLexicalForms() throws Exception {
        ObjectNode document = (ObjectNode) read("prepared_binary_v2.json");
        document.put("format_version", new BigDecimal("2.0"));
        ((ObjectNode) document.get("quantization_schema"))
                .put("nan_mode", new BigDecimal("1e0"));
        ((ObjectNode) document.get("trees").get(0).get("nodes").get(0))
                .put("split_bin_index", new BigDecimal("0.0"));
        Path artifact = Files.createTempFile("ctboost-exact-decimal", ".json");
        try {
            MAPPER.writeValue(artifact.toFile(), document);
            assertDoesNotThrow(() -> JsonPredictor.load(artifact));
        } finally {
            Files.deleteIfExists(artifact);
        }
    }

    @Test
    void cyclicTreesAreRejectedBeforeScoring() throws Exception {
        ObjectNode document = (ObjectNode) read("prepared_binary_v2.json");
        ObjectNode rootNode = (ObjectNode) document
                .get("trees").get(0).get("nodes").get(0);
        rootNode.put("left_child", 0);
        Path invalid = Files.createTempFile("ctboost-cycle", ".json");
        try {
            MAPPER.writeValue(invalid.toFile(), document);
            assertThrows(InvalidModelException.class, () -> JsonPredictor.load(invalid));
        } finally {
            Files.deleteIfExists(invalid);
        }
    }

    private static JsonNode read(String name) throws IOException {
        return MAPPER.readTree(fixture(name).toFile());
    }

    private static Path fixture(String name) {
        String override = System.getenv("CTBOOST_CONFORMANCE_DIR");
        Path directory = override == null || override.isBlank()
                ? Path.of("..", "..", "tests", "export_conformance")
                : Path.of(override);
        return directory.toAbsolutePath().normalize().resolve(name);
    }

    private static float[][] rows(JsonNode node) {
        float[][] result = new float[node.size()][];
        for (int row = 0; row < result.length; ++row) {
            JsonNode values = node.get(row);
            result[row] = new float[values.size()];
            for (int column = 0; column < result[row].length; ++column) {
                JsonNode value = values.get(column);
                result[row][column] = value.isTextual()
                        && value.textValue().equals("NaN")
                        ? Float.NaN
                        : value.floatValue();
            }
        }
        return result;
    }

    private static double[] doubles(JsonNode node) {
        double[] result = new double[node.size()];
        for (int index = 0; index < result.length; ++index) {
            result[index] = node.get(index).doubleValue();
        }
        return result;
    }
}
