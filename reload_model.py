# # hack to change model config from keras 2->3 compliant
# import h5py
# f = h5py.File("keras_model.h5", mode="r+")
# model_config_string = f.attrs.get("model_config")
# if model_config_string.find('"groups": 1,') != -1:
#     model_config_string = model_config_string.replace('"groups": 1,', '')
#     f.attrs.modify('model_config', model_config_string)
#     f.flush()
#     model_config_string = f.attrs.get("model_config")
#     assert model_config_string.find('"groups": 1,') == -1

# f.close()

import h5py

def fix_depthwise_layers(h5_file_path):
    with h5py.File(h5_file_path, 'r+') as f:
        model_config = f.attrs.get('model_config')
        
        # Sửa chỗ này
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        elif not isinstance(model_config, str):
            raise ValueError("Unexpected model_config type")
        
        # Tiếp tục xử lý
        model_config = model_config.replace('"groups": 1,', '')
        model_config = model_config.replace('"groups":1,', '')
        model_config = model_config.replace('"groups": 1', '')
        model_config = model_config.replace('"groups":1', '')
        
        f.attrs['model_config'] = model_config.encode('utf-8')

fix_depthwise_layers("keras_model.h5")
print("✅ Đã sửa file model thành công!")