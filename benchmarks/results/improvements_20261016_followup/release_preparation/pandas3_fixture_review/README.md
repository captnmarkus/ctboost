# pandas 3 fixture compatibility draft

Prepared 2026-10-16. This is a test-only, unexecuted patch draft. Neither the release checkout nor the original scientific source files were modified. The production dispatcher and all performance claims remain unchanged. `manifest.json` binds the original and proposed files; `fixture_changes.patch` is their unified diff.

The official current documentation is pandas 3.0.6, released September 17, 2026. [Release notes](https://pandas.pydata.org/docs/whatsnew/v3.0.6.html).

pandas 3 infers a string extension dtype by default. Explicit object dtype remains supported; the string dtype can use a NumPy object backend even without PyArrow, so omitting PyArrow does not restore the old dtype contract. [Official migration guide](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html).

Categorical construction preserves an object-typed pandas Index or Series, whereas a NumPy object array or Python sequence can still infer a more specific category dtype. Positive fixtures therefore specify `pd.Index(..., dtype=object)` for their category vocabulary. The native-code dispatcher additionally requires an object-typed column-name Index. [pandas 3.0 categorical construction note](https://pandas.pydata.org/docs/whatsnew/v3.0.0.html).

## Static findings and proposed coverage

| File | Draft change |
| --- | --- |
| `tests/test_native_categorical_codes.py` | Construct eligible object category and column Indexes explicitly for width/row boundary, codec/persistence, feature-name validation, legacy-native, mutating-name and late-vocabulary cases. Assert eligibility before checking feature-name or legacy fallback behavior. Add separate category-Index and column-Index `StringDtype(storage="python")` fallbacks, requiring `None` from the private dispatcher and exact public transform bits. |
| `tests/test_feature_pipeline_categorical_row.py` | Preserve intended positive category vocabulary dtype, including different category orders, Unicode, missing values, and public native-transform coverage. Add an extension-category fallback test that observes the original DataFrame conversion and checks exact cell types and public transform bits. Column-name dtype is not an eligibility requirement for this row shortcut. |
| `tests/test_categorical_codes_prototype.py` | Preserve the rejected historical prototype's test intent with explicit object category Indexes. Keep used invalid-Unicode values in an object Index too, so the exception comparison reaches the CTBoost codec instead of failing during pandas string construction. No prototype implementation change. |

The proposed tests retain positive-path assertions; they do not accept fallback as a substitute for testing the fast path. Unsupported extension-dtype cases deliberately fall back, with parity checked independently. No global pandas inference option is changed and pandas is not capped below version 3.

## Pending validation

No test module was imported or executed for this draft. The syntax-only AST check is recorded separately and is not a test result. After root reviews and applies the patch to the isolated release checkout, run the three focused files on pandas 2.3.3 and 3.0.6 using the validated native runtime. The pinned scientific CI job covers the historical exact stack; the ordinary latest-dependency CMake job must continue to cover pandas 3. This draft makes no new inference-speed claim for extension-string inputs.
