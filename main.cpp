#include "const.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <endian.h>

extern "C" {
    double NeuralNetwork(double image[N], double weights[NW], double label[O]);
    void NeuralNetwork_b(double image[N], double weights[NW], double weightsb[NW],
        double label[O], double NeuralNetworkb);
}

int main() {
    srand(10);
    size_t len=0;
    std::vector<double> images;
    std::vector<double> labels;
    {
        FILE* f = fopen("data/train-images.idx3-ubyte","rb");
        assert(f != NULL);
        int head[4];
        fread(head, sizeof(int), 4, f);
        for (size_t i=0;i<4;i++) head[i] = be32toh(head[i]);
        assert(head[2] == 28);
        assert(head[3] == 28);
        size_t total = head[1]*head[2]*head[3];
        len = head[1];
        std::vector<unsigned char> data;
        data.resize(total);
        images.resize(total);
        fread(data.data(), sizeof(char), total, f);
        for(size_t i = 0; i<total; i++) images[i] = (double) data[i] / 255;
        fclose(f);
    }
    {
        FILE* f = fopen("data/train-labels.idx1-ubyte","rb");
        assert(f != NULL);
        int head[2];
        fread(head, sizeof(int), 2, f);
        for (size_t i=0;i<2;i++) head[i] = be32toh(head[i]);
        assert(head[1] == len);
        size_t total = head[1];
        std::vector<unsigned char> data;
        data.resize(total);
        labels.resize(len*O);
        for(size_t i = 0; i<len*O; i++) labels[i] = 0;
        fread(data.data(), sizeof(char), total, f);
        for(size_t i = 0; i<total; i++) {
            int val = data[i];
            assert( val < O );
            labels[val + O*i] = 1;
        }
        fclose(f);
    }
    std::vector<double> weights;
    weights.resize(NW);
    for(size_t i = 0; i<weights.size(); i++) weights[i] = 2.0*rand()/RAND_MAX - 1.0;
    std::vector<double> weightsb;
    weightsb.resize(NW);

    double loss = 0;
    for (size_t i=0; i<len; i++) loss += NeuralNetwork(&images[N*i], weights.data(), &labels[O*i]);
    printf("loss: %lg\n", loss);
    for(size_t i=0; i<weightsb.size(); i++) weightsb[i] = 0;
    for(size_t i=0; i<len; i++) NeuralNetwork_b(&images[N*i], weights.data(), weightsb.data(), &labels[O*i], 1.0);
    for(size_t i=0; i<weightsb.size(); i++) printf("grad[%d]: %lg\n", (int) i, weightsb[i]);
    return 0;
}
