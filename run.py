from vllm import LLM, SamplingParams
import pandas as pd


model_type = "llama/"
template = "04"
model_name = "meta-llama/Llama-2-13b-chat-hf"
lang_pair = "en-de"
quantization = None


def main(max_model_len=1024, gpu_memory_util=0.9, quantization=None, temperature=0.8, top_p=0.95):
    data = pd.read_csv(model_type + lang_pair + "_vllm_t" + template + ".tsv", encoding="utf-8", sep="\t")
    prompts = data["final_prompt"].tolist()


    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=512)

    llm = LLM(model=model_name, gpu_memory_utilization=gpu_memory_util, max_model_len=max_model_len, dtype="auto", quantization=quantization) 

    outputs = llm.generate(prompts, sampling_params)

    with open(lang_pair.upper() + "_outputs_t" + template + "-" + model_name.split("/")[-1] + ".tsv", "w", encoding="utf-8") as f:
        f.write(f"prompt\tvllm_output\n")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            f.write(f"{prompt!r}\t{generated_text!r}\n")
    print("Done!")

if __name__ == "__main__":
    main(quantization=quantization)