import torch
from lungllm.models.gen_model import GenModel
from lungllm.eval.eval_faithfulness import load_clips, factscore
clips = load_clips("data/splits/icbhi_official_test.parquet", 15)
specs = {"sft":"checkpoints/gen/sft_rich.pt","grpo":"checkpoints/gen/grpo.pt"}
models = {}
for label,path in specs.items():
    m = GenModel("Qwen/Qwen2.5-3B-Instruct",
                 encoder_ckpt="checkpoints/aligned/ast_aligned_clean.pt")
    m.load_state_dict(torch.load(path,map_location="cpu")["model"], strict=False)
    m.to("cuda").eval(); models[label]=m
tok = models["sft"].llm.tok
pr = tok(["Auscultation"], return_tensors="pt")
for i,(w,ev) in enumerate(clips):
    print(f"\n=== clip {i} true_event={ev} ===")
    for label,m in models.items():
        with torch.no_grad():
            gen = m.generate([w], pr["input_ids"].to("cuda"),
                             pr["attention_mask"].to("cuda"), max_new_tokens=24)
        rep = gen[0] if gen else ""
        print(f"[{label}] fs={factscore(rep,ev):.2f} | {rep!r}")
