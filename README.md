# federated-learning

## 安裝環境

`pip install tensorflow flwr==0.17.0`

> python Version 3.7 以上

## 先執行 server.py

`python server.py`

## 執行 client.py

`python client.py --partition <client-id>`

> 需要執行三個 client,client-id = 0,1,2

## 需要更改的地方(有附上連結可點)

- [sever](https://github.com/Andy106084/federated-learning/blob/main/server.py#L18) 和 [client](https://github.com/Andy106084/federated-learning/blob/main/client.py#L23) 的 def CNN_Model(input_shape, number_classes) 需要改為自己的 model
- [client](https://github.com/Andy106084/federated-learning/blob/main/client.py#L43) 的 def load_partition(idx: int) 需要改為自己的 dataset
- [server](https://github.com/Andy106084/federated-learning/blob/main/server.py#L98) get_eval_fn(model)裡面的 dataset 需要改為自己的 dataset
  - get_eval_fn 用來 update client 給的 weight
- [server](https://github.com/Andy106084/federated-learning/blob/main/server.py#L69) fit_config(rnd: int)裡面的 local_epochs 需要更改成自己需要的 epoch 數
  - config={"num_rounds": 2} 會影響到 epoch 次數
