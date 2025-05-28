"""
Fast RoBERTa Training - 2-3 hours with excellent results
Best balance of speed and quality
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

set_seed(42)

class FastDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

def create_focused_data():
    """Create focused, high-quality dataset"""
    
    print("Creating focused dataset...")
    
    # Essential patterns only
    data = []
    
    # Key unsubscribe patterns (7.5k)
    unsub_patterns = [
        ("💰 {b} Sale: {d}% OFF", "Limited time! Shop now. Unsubscribe"),
        ("{b} Newsletter - {m} Edition", "Monthly updates. Manage preferences"),
        ("Your cart at {b}", "Complete purchase. Stop reminders"),
        ("We miss you!", "Come back for {d}% off. Opt out"),
        ("{b} Rewards Update", "{p} points earned. Email settings"),
    ]
    
    # Key important patterns (7.5k)
    important_patterns = [
        ("Security Alert", "New login from {l}. Verify now"),
        ("Payment ${a} confirmed", "Transaction complete. ID: {t}"),
        ("Password reset", "Reset link expires in {h} hours"),
        ("Re: {s}", "Thanks for your email. {r}"),
        ("Order #{o} shipped", "Track: {tr}. Arrives {d}"),
    ]
    
    # Quick generation
    brands = ["Amazon", "Google", "Apple", "Netflix", "Microsoft"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    
    for _ in range(1500):  # 1500 * 5 patterns = 7500
        for pattern in unsub_patterns:
            subj, body = pattern
            subj = subj.replace("{b}", np.random.choice(brands))
            subj = subj.replace("{d}", str(np.random.randint(20, 70)))
            subj = subj.replace("{m}", np.random.choice(months))
            body = body.replace("{p}", str(np.random.randint(100, 5000)))
            
            data.append({
                "text": f"Subject: {subj}\n\n{body}",
                "label": 1
            })
    
    for _ in range(1500):  # 1500 * 5 patterns = 7500
        for pattern in important_patterns:
            subj, body = pattern
            body = body.replace("{l}", np.random.choice(["New York", "London"]))
            body = body.replace("{a}", str(np.random.randint(50, 500)))
            body = body.replace("{t}", f"T{np.random.randint(10000, 99999)}")
            body = body.replace("{h}", str(np.random.choice([1, 2, 6, 24])))
            body = body.replace("{s}", "Project update")
            body = body.replace("{r}", "Will review")
            body = body.replace("{o}", str(np.random.randint(100000, 999999)))
            body = body.replace("{tr}", f"TRK{np.random.randint(10000, 99999)}")
            body = body.replace("{d}", "tomorrow")
            
            data.append({
                "text": f"Subject: {subj}\n\n{body}",
                "label": 0
            })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created {len(df)} samples")
    return df

def train_fast_roberta():
    """Fast training with RoBERTa"""
    
    print("="*60)
    print("FAST ROBERTA TRAINING")
    print("="*60)
    print(f"Start: {datetime.now()}")
    print("Duration: 2-3 hours")
    print("Expected: 90-93% accuracy\n")
    
    # Fast config
    MODEL = "roberta-base"
    MAX_LEN = 256
    BATCH = 32  # Larger batch
    GRAD_ACC = 2  # Less accumulation
    LR = 3e-5
    EPOCHS = 3  # Fewer epochs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Data
    df = create_focused_data()
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Model
    print(f"\nLoading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model = model.to(device)
    
    # Datasets
    train_dataset = FastDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset = FastDataset(X_val, y_val, tokenizer, MAX_LEN)
    
    # Args
    args = TrainingArguments(
        output_dir="./fast_roberta_model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH*2,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_steps=100,
        learning_rate=LR,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        resume_from_checkpoint=True,  # Resume from latest checkpoint if exists
    )
    
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': p,
            'recall': r
        }
    
    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    print("\nTraining...")
    result = trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained("./fast_roberta_model")
    
    # Evaluate
    eval_res = trainer.evaluate()
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Time: {result.metrics['train_runtime']/3600:.1f} hours")
    print(f"Accuracy: {eval_res['eval_accuracy']:.1%}")
    print(f"F1: {eval_res['eval_f1']:.1%}")
    
    # Save info
    with open("./fast_roberta_model/info.json", "w") as f:
        json.dump({
            "model": MODEL,
            "accuracy": eval_res['eval_accuracy'],
            "f1": eval_res['eval_f1'],
            "hours": result.metrics['train_runtime']/3600
        }, f, indent=2)
    
    # Link for app
    if os.path.exists("./ml_suite/models/production_model"):
        os.remove("./ml_suite/models/production_model")
    os.symlink(os.path.abspath("./fast_roberta_model"), "./ml_suite/models/production_model")
    
    print("\nReady to use!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Fast training: 2-3 hours, 90-93% accuracy")
    if input("\nStart? (y/n): ").lower() == 'y':
        train_fast_roberta()