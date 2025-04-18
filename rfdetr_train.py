from rfdetr import RFDETRBase
import pandas as pd

dataset_path = "./data/thera_bihar_to_test_bihar"
output_path = "./runs/thera_bihar_to_test_bihar"


model = RFDETRBase()
history = []
def callback2(data):
    history.append(data)
model.callbacks["on_fit_epoch_end"].append(callback2)

model.train(
    dataset_dir=dataset_path,
    epochs=50,
    batch_size=32,
    grad_accum_steps=1,
    lr=1e-4,
    output_dir=output_path,
    # resume="runs/exp1/checkpoint.pth",
    tensorboard=True,
)


pd.DataFrame(history).to_csv(f"{output_path}/history.csv", index=False)



# source .venv/bin/activate
# export CUDA_VISIBLE_DEVICES=1
# nohup python rfdetr_train.py > ./logs/thera_bihar_to_test_bihar.log 2>&1 &
# tensorboard --logdir ./runs/thera_bihar_to_test_bihar
