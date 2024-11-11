from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

class PromptInjection:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection")
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def detect_prompt_injection(self, text):           
        if len(text) > 512:
            return {
                "answer": "Prompt greater than 512 tokens. Please provide a shorter prompt.",
                "source_document": {
                    "content": "null",
                    "metadata": {"urls": "null"},
            }
          }
        
        prompt_injection = self.classifier(text)
        print("Prompt Injection Detection: ", prompt_injection)
        
        if not prompt_injection[0]['label'] == "SAFE" :
          return  {
                "answer": "Security issues detected with the prompt, please reformat your question otherwise access will be revoked!",
                "source_document": {
                    "content": "null",
                    "metadata": {"urls": "null"},
            }
          }
        else:
            return "prompt_safe" 

