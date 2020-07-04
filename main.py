from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import Model,Sequential
import numpy as np
import time
import os

np.random.seed(10)
classes = ["unknown","bed","cat","follow","forward","go","left","no","on","off","right"]
classes_to_idx = {classes[i]:i for i in range(len(classes))}
index = "16_32kernel_prune"
save_dir = "./model/{}/".format(index)
data_path = "./data/"


def load_data(data_path):
    train_folder = data_path + "MFCC_train/"
    test_folder = data_path + "MFCC_test/"
    train_path = "MFCC_train/train_labels.txt"
    test_path = "MFCC_test/test_labels.txt"
    #把所有mfcc和label分别放在一个npy文件中，如果没有，则从每个文件夹中读取然后生成npy文件
    try:
        print(os.getcwd())
        train_data = np.load(train_folder+"all_train_data.npy")
        train_label = np.load(train_folder+"all_train_label.npy")
        test_data = np.load(test_folder+"all_test_data.npy")
        test_label = np.load(test_folder+"all_test_label.npy")
    except:
        print("data not generated,load from folder and generate all data in a npy file")
        train_data, train_label = load(data_path + train_path,train_folder,2000)
        test_data, test_label = load(data_path + test_path, test_folder,200)
        np.save(train_folder+"all_train_data.npy",train_data)
        np.save(train_folder+"all_train_label.npy",train_label)
        np.save(test_folder+"all_test_data.npy",test_data)
        np.save(test_folder+"all_test_label.npy",test_label)
    #转为四维张量
    train_data = train_data[:,:,:,np.newaxis].astype("float32")
    test_data = test_data[:,:,:,np.newaxis].astype("float32")
    train_label = train_label.astype("int8")
    test_label = test_label.astype("int8")
    #打乱数据
    np.random.seed(10)
    np.random.shuffle(train_data)
    np.random.seed(10)
    np.random.shuffle(train_label)
    np.random.seed(10)
    np.random.shuffle(test_data)
    np.random.seed(10)
    np.random.shuffle(test_label)
    #标准化处理
    mean_train = np.mean(train_data)
    std_train = np.std(train_data)
    train_data = (train_data-mean_train)/std_train

    mean_test = np.mean(test_data)
    std_test = np.std(test_data)
    test_data = (test_data-mean_test)/std_test

    return train_data, train_label,test_data, test_label

def load(data_path,folder,num_of_unknown):
    #创建空数组然后append或者创建空列表append然后一次性转换为数组
    #空数组匹配数据的维度
    count = 0
    datas = []
    labels = []
    #load train data
    with open(data_path) as f:
        for line in f:
            path = line.split(' ')[0]
            data = np.load(folder+path)
            label = line.split(' ')[1].strip('/n')
            label = int(label)
            if label == 0 :
                if count<num_of_unknown:
                    datas.append(data)
                    labels.append(label)
                    count += 1
            else:
                datas.append(data)
                labels.append(label)
    datas = np.array(datas)
    labels = np.array(labels)
    return  datas,labels
    
class fourlayerCNN(Model):
    def __init__(self,num_of_class = 11):
        super(fourlayerCNN,self).__init__()
        self.conv1 = layers.Conv2D(16,3,activation = 'relu')
        self.conv2 = layers.Conv2D(16,3,activation = 'relu')
        self.maxpooling1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = layers.Conv2D(32,3,activation = 'relu')
        self.conv4 = layers.Conv2D(32,3,activation = 'relu')
        self.maxpooling2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.d2 = layers.Dense(num_of_class, activation='softmax')
    
    def call(self,inputs):
        activations = []
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(self.maxpooling1(x2))
        x4 = self.conv4(x3)
        
        outputs = self.d2(self.flatten(self.maxpooling2(x4)))
        return outputs


class Mymodel():
    def __init__(self):
        self.input_shape = (40,44,1)
        self.model = self.get_model("functional")

    def get_model(self,name):
        if name is "functional":
            inputs = keras.Input(shape = self.input_shape)
            x1 = layers.Conv2D(4, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')(inputs)
            x2 = layers.Conv2D(4, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')(x1)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x2)
            x3 = layers.Conv2D(8, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')(x)
            x4 = layers.Conv2D(4, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')(x3)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x4)
            x = layers.Flatten()(x)
            prediction = layers.Dense(11,activation = "softmax")(x)
            return keras.Model(inputs = inputs, outputs = prediction)
        elif name is "subclass":
            return fourlayerCNN()


    def info(self):
        for i in range(10):
            print("Training size of {} is {}".format(classes[i],train_data[train_label==i].shape[0]))
        self.model.summary()
        

    def train(self,train_data,train_label,model_name = None):

        def scheduler(epoch,lr):
            if epoch <5:
                return lr
            elif epoch <7:
                return lr/10
            else:
                return lr/1000 

        callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 1e-2,patience = 2, verbose = 1)]
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-4),
        #loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        history = self.model.fit(train_data,train_label,batch_size = 16,epochs = 10,callbacks = callback,validation_split = 0.1)
 
        #save model
        model_name = "original_model.h5"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        try:
            self.model.save(save_dir+model_name)
        except:
            model_name = "original_model"
            self.model.save_weights(save_dir + model_name,save_format = 'tf')
        convert2tflite(self.model,train_data[:10])


    def test(self,test_data,test_label, test_model = None):
        if test_model == None:
            test_model = self.model
        for i in range(10):
            print("size of {} is {}.Test accuracy are as follow".format(classes[i],test_data[test_label==i].shape[0]))
            res = test_model.evaluate(test_data[test_label==i],test_label[test_label==i])
        print("all Test accuracy are as follow")
        res = test_model.evaluate(test_data,test_label)


    def get_weights(self):
        weights = self.model.weights
        tf.print(weights[0])
        return weights


    def prune(self, target_data):
        selected_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
        activations,feature_extractor = self.get_activation(selected_layers,target_data)
        layer_index = self.APOZ(activations)
        prune_model = self.replace_layer(selected_layers,layer_index,feature_extractor)


    def get_activation(self,selected_layers,target_data):
        activations = []
        layers = [layer.name for layer in self.model.layers]
        weights = [self.model.get_layer(name).weights for name in selected_layers ]
        outputs = [self.model.get_layer(name).output for name in selected_layers]
        feature_extractor = keras.Model(self.model.inputs,outputs)
        ###可再加个activation selection的功能####

        ########################################
        activations = feature_extractor(target_data)
        print(layers)
        return activations,feature_extractor

    def APOZ(self,activations):
        threshold = 50
        layers_indexs = []
        for activation in activations:
            indexs = []
            activation = tf.transpose(activation, perm = [3,0,1,2])
            for _,features in enumerate(activation):
                size = tf.size(features).numpy()
                num_zeros = tf.size(features[features == 0]).numpy()
                APOZ = 100 * num_zeros/size
                if APOZ > threshold:
                    indexs.append(0)
                else:
                    indexs.append(1)
            layers_indexs.append(indexs)
        return layers_indexs
                

    def replace_layer(self,selected_layers,layer_index,feature_extractor):
        #1)先遍历原模型，把conv曾按照apoz结果设置通
        new_model = keras.Sequential()
        for layer in self.model.layers[:len(self.model.layers)-1]:
            if layer.name in selected_layers:
                index = selected_layers.index(layer.name)
                channels = layer_index[index]
                channels = np.array(channels)
                num_channels = channels.sum()
                new_layer = keras.layers.Conv2D(
                    num_channels, kernel_size=(3, 3), activation="relu",strides = (1,1),padding = 'same')
                new_model.add(new_layer)
            else:
                new_model.add(layer)
            new_conv_layer = [layer.name for layer in new_model.layers if layer.name.startswith('conv') ]
        print("new_model: {}".format(new_conv_layer)) 
        print("new_model_all: {}".format([layer.name for layer in new_model.layers]))
        

        #2)然后再根据apoz结果得到想要的weight，在进行权重初始化
        #得到一层中index为1的layer
        old_weights = [self.model.get_layer(name).weights for name in selected_layers ]
        for i,name in enumerate(selected_layers):
            ###更新当前layer的权值
            current_layer = new_model.get_layer(new_conv_layer[i])
            current_old_weights = old_weights[i][0]
            bias = old_weights[i][1]
            current_old_weights = tf.transpose(current_old_weights,perm = [3,0,1,2])
            index = np.array(layer_index[i])
            new_weights = current_old_weights[index == 1]
            new_weights = tf.transpose(new_weights,perm = [1,2,3,0])
            bias = bias[index == 1]
            current_layer.weights[0].assign(new_weights)
            current_layer.weights[1].assign(bias)
            ###如果有下一个 conv layer, 也进行剪裁
            if i < len(new_conv_layer)-1:
                next_layer = new_model.get_layer(new_conv_layer[i+1])
                next_weights = tf.transpose(old_weights[i+1][0],perm = [2,0,1,3])
                next_weights = next_weights[index == 1]
                next_weights = tf.transpose(next_weights,perm =[1,2,0,3])
                old_weights[i+1][0] = next_weights
            
        self.new_model = new_model
        
    def retrain_model(self,train_data,train_label,test_data,test_label, istest = False):         
        #先增加全连接层，然后根据输入的数据进行retrain
        if istest == True:
            self.new_model.add(self.model.get_layer('dense'))
            self.test(test_data = test_data,test_label=test_label,test_model = self.new_model)
            self.new_model.pop(self.new_model.get_layer('dense'))        
        fully_connected = layers.Dense(1,activation = "softmax")
        #obtain retraining data
        ##positive samples
        
        
        #retrain process




def convert2tflite(model,data):
    model_name = "origin_model.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # def representative_dataset_generator():
    #     for value in data:
    #         yield [value[np.newaxis]]
    
    # converter.representative_dataset = representative_dataset_generator
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.target_spec.supported_types = [tf.float32]
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    open( save_dir + model_name, "wb").write(tflite_quant_model)


def test_tflite(data):
    tflite_model = tf.lite.Interpreter('./model/origin_model.tflite')
    tflite_model.allocate_tensors()
    details = tflite_model.get_input_details()[0]
    input_index = tflite_model.get_input_details()[0]['index']
    output_index = tflite_model.get_output_details()[0]['index']
    model_prediction = []


    for value in data:
        tensor = tf.convert_to_tensor([value],dtype = np.float32)
        tflite_model.set_tensor(input_index,tensor)
        tflite_model.invoke()
        model_prediction.append(tflite_model.get_tensor(output_index)[0])


if __name__ == "__main__":
    time1 = time.time()
    train_data, train_label,test_data, test_label = load_data(data_path = data_path )
    time2 = time.time()
    print("load time is {}".format(time2-time1))
    isload_model = True
    if isload_model is True:
        Prune_model = Mymodel()
        Prune_model.model = tf.keras.models.load_model(save_dir+'original_model.h5'.format(index))
        convert2tflite(Prune_model.model,train_data[:10])
    else:
        Prune_model = Mymodel()
        Prune_model.train(train_data,train_label)
        Prune_model.info()

    test_tflite(train_data[:10])
    #model.get_weights()
    Prune_model.prune(test_data[test_label == 4])
    #Prune_model.tests(test_data,test_label)
    Prune_model.retrain_model(train_data = train_data,train_label = train_label,
                                test_data=test_data,test_label= test_label)

    #activations = model.get_activations(train_data[:10])
    #print(len(activations))
    #tf.print(activations)
    pass