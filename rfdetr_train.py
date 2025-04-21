from rfdetr import RFDETRBase, RFDETRLarge
import pandas as pd

dataset_path = "./data/haryana_to_test_bihar"
output_path = "./runs/haryana_to_test_bihar"

# model = RFDETRBase()
model = RFDETRLarge()
history = []
def callback2(data):
    history.append(data)
model.callbacks["on_fit_epoch_end"].append(callback2)

model.train(
    dataset_dir=dataset_path,
    epochs=100,
    # batch_size=32,
    batch_size=16,
    grad_accum_steps=1,
    lr=2e-4,
    output_dir=output_path,
    # resume="runs/haryana_to_test_bihar/checkpoint0079.pth",
    tensorboard=True,
    # early_stopping=True
)


pd.DataFrame(history).to_csv(f"{output_path}/history.csv", index=False)


# source .venv/bin/activate
# export CUDA_VISIBLE_DEVICES=3
# nohup python rfdetr_train.py > ./logs/rfdetr_large/haryana_to_test_bihar.log 2>&1 &
# tensorboard --logdir ./runs/haryana_to_test_bihar
