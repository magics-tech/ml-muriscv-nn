# Custom Model
In `model.py` we have defined a custom model to mimic the VWW input and output. Running this python script will result in `toy_example.tflite` that can be used to replace `vww.tflite` in the integration folder.

Note that the model has to be replaced, prior to calling `setup_tvm.sh`, which in its turn, must be called prior to building the c-sources.