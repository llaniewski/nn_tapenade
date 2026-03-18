#include <math.h>
#include <assert.h>
#include "const.h"

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

void NeuralNetwork(double image[N], double weights[NW], double layer3[O]) {
    int widx = 0;
    double layer1m[M];
    for (int i=0;i<M;i++) {
        double sum = 0;
        for (int j=0;j<N;j++) {
            sum += weights[widx] * image[j];
            widx++;
        }
        sum += weights[widx];
        widx++;
        layer1m[i] = sum;
    }
    double layer1[M];
    for (int i=0;i<M;i++) {
        layer1[i] = sigmoid(layer1m[i]);
    }
    double layer2m[M];
    for (int i=0;i<M;i++) {
        double sum = 0;
        for (int j=0;j<M;j++) {
            sum += weights[widx] * layer1[j];
            widx++;
        }
        sum += weights[widx];
        widx++;
        layer2m[i] = sum;
    }
    double layer2[M];
    for (int i=0;i<M;i++) {
        layer2[i] = sigmoid(layer2m[i]);
    }
    double layer3m[O];
    for (int i=0;i<O;i++) {
        double sum = 0;
        for (int j=0;j<M;j++) {
            sum += weights[widx] * layer2[j];
            widx++;
        }
        sum += weights[widx];
        widx++;
        layer3m[i] = sum;
    }
    for (int i=0;i<O;i++) {
        layer3[i] = sigmoid(layer3m[i]);
    }
}

double NeuralNetworkLoss(double image[N], double weights[NW], double label[O]) {
    double layer3[O];
    NeuralNetwork(image, weights, layer3);
    double loss = 0;
    for (int i=0;i<O;i++) {
        loss += (layer3[i]-label[i])*(layer3[i]-label[i]);
        // loss += -label[i]*log(layer3[i]);
    }
    return loss;
}
