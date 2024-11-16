# federated-learning
## 安裝環境 
`pip install tensorflow flwr==0.17.0`
>python Version 3.7以上
## 先執行server.py
`python server.py`
## 執行client.py
`python client.py --partition <client-id>`
> 需要執行三個client,client-id = 0,1,2
## 需要更改的地方
* 第18行def CNN_Model(input_shape, number_classes)需要改為自己的model1`
* 第98行get_eval_fn(model)裡面的dataset需要改為自己的dataset`
* 第69行fit_config(rnd: int)裡面的local_epochs需要更改成自己需要的epoch數`



