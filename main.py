import torch
import os
from preprocess import load_and_preprocess, create_dataloaders
from train_eval_model import MLPClassifier, train, evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = input("Please enter the CSV file path: ")
    label_column = input("Please enter the name of the target column: ")

    x_train, y_train, x_test, y_test = load_and_preprocess(csv_path, label_column)
    input_dim = x_train.shape[1]
    output_dim = len(torch.unique(y_train))

    model = MLPClassifier(input_dim, output_dim).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Model created. Starting training...")
    epochs = int(input("Please enter the number of training epochs: "))

    train_loader, test_loader = create_dataloaders(x_train, y_train, x_test, y_test)
    train(model, train_loader, loss_fn, optimizer, epochs, device)
    evaluate(model, test_loader, device)

    save_path = input("Enter path to save the model (e.g., models/model.pt): ")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
