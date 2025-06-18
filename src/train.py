import wandb
from tqdm import tqdm
from src.evaluate import evaluate
import torch

def train_model(model, optimizer, configs, loaders):

    # Login wandb
    wandb.login()

    # Init Wandb for tracking training phase
    wandb.init(
        project=configs["project"],
        name=configs["name"],
        config=configs
    )

    # Log gradient of parameter
    wandb.watch(model, log="all")

    # Save model checkpoint by best F1
    best_val_f1 = 0.0

    # Training Loop
    for epoch in range(1, configs["epochs"] + 1):
        model.train()
        total_loss = 0.0

        # Create progress bar
        train_bar = tqdm(loaders['train'], desc=f"Train Epoch {epoch}/{configs['epochs']}")

        for batch_idx, (x, y, _) in enumerate(train_bar, start=1):
            mask = (y != -1)
            loss = model(x, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_bar.set_postfix(batch_loss=loss.item(), avg_loss=total_loss / batch_idx)
        
        # Evaluate model after each epoch
        avg_train_loss = total_loss / len(loaders['train'])
        train_precision, train_recall, train_f1, train_acc, _, _ = evaluate(model, loaders['train'], count_loss=False)
        val_precision, val_recall, val_f1, val_acc, avg_val_loss, _= evaluate(model, loaders['val'], count_loss=True)
        
        # Log metric for train and val set
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_f1={train_f1:.4f}, val_loss={avg_val_loss:.4f}, val_f1={val_f1:.4f}")
        wandb.log({

            "epoch": epoch,

            # Group: Training metrics
            "Train/Loss": avg_train_loss,
            "Train/Precision": train_precision,
            "Train/Recall": train_recall,
            "Train/F1": train_f1,
            "Train/Accuracy": train_acc,
            
            # Group: Validation metrics
            "Val/Loss": avg_val_loss,
            "Val/Precision": val_precision,
            "Val/Recall": val_recall,
            "Val/F1": val_f1,
            "Val/Accuracy": val_acc
        })

        # Save best model based on val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = f"./models/best_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            wandb.save(ckpt_path)
            print(f"Saved imporved model to {ckpt_path}")
        
        print()
    
    # Load best model before test
    print(f"Loading best model from {ckpt_path} for final evaluation...")
    model.load_state_dict(torch.load(ckpt_path))
    print("Done \n")

        
    # Log metric for test set
    print("Evaluation on test set ...")
    test_precision, test_recall, test_f1, test_acc, avg_test_loss, report = evaluate(model, loaders['test'], count_loss=True, report=True)
    wandb.log({
        "Test/Loss": avg_test_loss,
        "Test/Precision": test_precision,
        "Test/Recall": test_recall,
        "Test/F1": test_f1,
        "Test/Accuracy": test_acc,
    })
    print(f"Test_loss={avg_test_loss:.4f}, Test_f1={test_f1:.4f}")
    print(report)

    # Finish W&B run
    wandb.finish()