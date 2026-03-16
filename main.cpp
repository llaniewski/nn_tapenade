#include <stdio.h>
#include "const.h"

extern "C" {
    double NeuralNetwork(double image[N], double weights[NW], double label[O]);
    void NeuralNetwork_b(double image[N], double weights[NW], double weightsb[NW],
        double label[O], double NeuralNetworkb);
}

int main() {
    return 0;
}
