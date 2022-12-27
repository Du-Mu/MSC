import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

def cal_dist(img1, img2):
    '''
    get euclidean distance of img1 and img2
    '''
    dist = np.sqrt( \
    np.sum(np.square(img1 - img2)))

    return dist


def get_dists(img_test, img_train):
    '''
    get all euclidean distance of test_set and train_set
    img_test: numpy array of img_test
    img_train: numpy array of img_train
    '''
    test_num = img_test.shape[0]
    train_num = img_train.shape[0]
    dists = np.zeros([test_num,train_num])
    for i in range(test_num):
        for j in range(train_num):
            dists[i,j] = cal_dist(img_test[i],img_train[j])
    
    return dists

def get_labels(img_test, img_train, train_label, dists, k):
    '''
    get the nerest label
    '''
    test_num = img_test.shape[0]

    test_labels_pre = np.zeros(test_num)
    for i in range(test_num):
        min_k_arr = np.argsort(dists[i])[:k]  
        train_labels_k = train_label[min_k_arr]
        test_labels_pre[i] = np.argmax(np.bincount(train_labels_k))
    
    return test_labels_pre


def KNN(img_test, img_train, train_label, test_label, k):
    dists = get_dists(img_test, img_train)
    test_labels_pre = get_labels(img_test, img_train, train_label, dists, k)

    correct_num = np.sum(test_labels_pre == test_label)
    discorrect_num = np.sum(test_labels_pre != test_label)
    accuracy = correct_num/(correct_num+discorrect_num)
    print(f"K={k}:  Correct:{correct_num}\t Discorrect:{discorrect_num}\t Accuracy:{accuracy}\n")
    print("--------------------------------------------------------------------------------------")

    return accuracy

def load_all_images(test_num, train_num, classes):
    test_path = './classed_dataset/test_data/'
    train_path = './classed_dataset/trainning_data/'
    # load test_data
    img_test = np.zeros([test_num,64,64])
    test_labels = np.zeros(test_num,dtype='uint8')
    label_num = 0
    count = 0
    for i in classes:
        images = os.listdir(test_path+i)
        for j in images:
            image_path = test_path+i+'/'+j
            img = Image.open(image_path)
            np_img = np.array(img)
            img_test[count] = np_img/255
            test_labels[count] = label_num
            count+=1
        label_num+=1

    img_train = np.zeros([train_num,64,64])
    train_labels = np.zeros(train_num,dtype='uint8')
    label_num = 0
    count = 0
    for i in classes:
        images = os.listdir(train_path+i)
        for j in images:
            image_path = train_path+i+'/'+j
            img = Image.open(image_path)
            np_img = np.array(img)
            img_train[count] = np_img/255
            train_labels[count] = label_num
            count+=1
        label_num+=1

    return img_test,test_labels,img_train,train_labels

def main():
    classes = os.listdir('./classed_dataset/test_data')
    test_num = 0
    for i in classes:
        images = os.listdir('./classed_dataset/test_data/'+i)
        test_num+=len(images)
    train_num = 0
    for i in classes:
        images = os.listdir('./classed_dataset/trainning_data/'+i)
        train_num+=len(images)

    img_test,test_labels,img_train,train_labels = load_all_images(test_num, train_num, classes)
    k = []
    accuracys = []
    for i in range(1,11):
        accuracys.append(KNN(img_test=img_test, img_train=img_train, test_label=test_labels, train_label=train_labels, k=i))
        k.append(i)
    plt.plot(k, accuracys)
    plt.ylabel('accuracy')
    plt.xlabel('k')
    plt.show()


if __name__ == "__main__":
    main()