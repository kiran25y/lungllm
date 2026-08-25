import importlib, glob, ast, pytest
FILES = glob.glob("src/lungllm/**/*.py", recursive=True)
@pytest.mark.parametrize("f", FILES)
def test_parses(f):
    ast.parse(open(f).read())
TORCHFREE = ["lungllm","lungllm.data.ingest","lungllm.data.splits","lungllm.data.splits_v2",
             "lungllm.data.build_text","lungllm.data.lungmix","lungllm.eval.faithfulness",
             "lungllm.eval.evaluate_v2","lungllm.rag.build_reports"]
@pytest.mark.parametrize("m", TORCHFREE)
def test_imports(m):
    importlib.import_module(m)
