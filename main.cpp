#include "const.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <string>
#include <endian.h>
#include <nlopt.h>
#include <algorithm>
#include <random>

extern "C" {
    double NeuralNetwork(double image[N], const double weights[NW], double layer3[O]);
    double NeuralNetworkLoss(double image[N], const double weights[NW], double label[O]);
    void NeuralNetworkLoss_b(double image[N], const double weights[NW], double weightsb[NW],
        double label[O], double NeuralNetworkb);
}

struct image_set {
    size_t len=0;
    std::vector<double> images;
    std::vector<int> labels;
    std::vector<double> label_dists;
    std::vector<size_t> idx;
    void read(std::string name) {
        {
            FILE* f = fopen((name+"-images.idx3-ubyte").c_str(),"rb");
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
            FILE* f = fopen((name+"-labels.idx1-ubyte").c_str(),"rb");
            assert(f != NULL);
            int head[2];
            fread(head, sizeof(int), 2, f);
            for (size_t i=0;i<2;i++) head[i] = be32toh(head[i]);
            assert(head[1] == len);
            size_t total = head[1];
            std::vector<unsigned char> data;
            data.resize(total);
            label_dists.resize(len*O);
            labels.resize(len);
            for(size_t i = 0; i<len*O; i++) label_dists[i] = 0;
            fread(data.data(), sizeof(char), total, f);
            for(size_t i = 0; i<total; i++) {
                int val = data[i];
                assert( val < O );
                labels[i] = val;
                label_dists[val + O*i] = 1;
            }
            fclose(f);
        }
        idx.resize(len);
        for(size_t i = 0; i<len; i++) idx[i] = i;
    }
    double avg_loss(const double* weights, int len0) {
        double loss = 0;
        double hit = 0;
        for (size_t i=0; i<len0; i++) {
            double output[O];
            size_t k = idx[i];
            NeuralNetwork(&images[N*k], weights, output);
            int label = labels[k];
            int imax = 0;
            double loss0 = 0;
            for (int j=0; j<O; j++) {
                if (output[j] > output[imax]) imax = j;
                double dist_val = label_dists[O*k + j];
                dist_val = dist_val - output[j];
                loss0 += dist_val*dist_val;
                // loss0 += -dist_val[i]*log(output[i]);
            }
            if (imax == label) hit++;
            loss += loss0;
            // loss += NeuralNetworkLoss(&images[N*i], weights, &label_dists[O*i]);
        }
        loss = loss / len0;
        hit = hit / len0;
        printf("loss: %10lg, hits: %.0lf%%\n", loss, hit*100);
        return loss;
    }
    void avg_loss_grad(const double* weights, double* weightsb, int len0) {
        double w = 1.0/len0;
        for(size_t i=0; i<NW; i++) weightsb[i] = 0;
        for(size_t i=0; i<len0; i++) {
            size_t k = idx[i];
            NeuralNetworkLoss_b(&images[N*k], weights, weightsb, &label_dists[O*k], w);
        }
    }
};

image_set train;
image_set test;
std::random_device rd;
std::mt19937 rnd(rd());

int main() {
    srand(10);
    train.read("data/fashion/train");
    test.read("data/fashion/t10k");

    std::vector<double> weights;
    weights.resize(NW);
    for(size_t i = 0; i<weights.size(); i++) weights[i] = 2.0*rand()/RAND_MAX - 1.0;
    std::vector<double> weightsb;
    weightsb.resize(NW);

    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, NW);
    nlopt_result opt_res;
    opt_res = nlopt_set_min_objective(
        opt,
        [](unsigned n, const double* x, double* grad, void* f_data) -> double {
            std::shuffle(train.idx.begin(), train.idx.end(), rnd);
            size_t len = train.len;
            printf("train: ");
            double loss = train.avg_loss(x,len);
            printf("test:  ");
            test.avg_loss(x,test.len);
 
 
 
            train.avg_loss_grad(x,grad, len);
            return loss;
        },
        NULL
    );
    opt_res = nlopt_set_maxeval(opt, 2000);
    double obj;
    opt_res = nlopt_optimize(opt, weights.data(), &obj);
    printf("obj: %lg", obj);
    return 0;
}
