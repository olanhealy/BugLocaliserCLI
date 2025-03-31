import torch.nn as nn
from transformers import RobertaModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Custom model which I used while training 
class CustomRobertaClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.roberta.resize_token_embeddings(config.vocab_size)
        for name, param in self.roberta.named_parameters():
            if "embeddings" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
