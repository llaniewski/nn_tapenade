#include "const.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <string>
#include <endian.h>
#include <nlopt.h>

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
    }
    double avg_loss(const double* weights) {
        double loss = 0;
        double hit = 0;
        for (size_t i=0; i<len; i++) {
            double output[O];
            NeuralNetwork(&images[N*i], weights, output);
            int label = labels[i];
            int imax = 0;
            double loss0 = 0;
            for (int j=0; j<O; j++) {
                if (output[j] > output[imax]) imax = j;
                double dist_val = label_dists[O*i + j];
                dist_val = dist_val - output[j];
                loss0 += dist_val*dist_val;
            }
            if (imax == label) hit++;
            loss += loss0;
            // loss += NeuralNetworkLoss(&images[N*i], weights, &label_dists[O*i]);
        }
        loss = loss / len;
        hit = hit / len;
        printf("loss: %10lg, hits: %.0lf%%\n", loss, hit*100);
        return loss;
    }
    void avg_loss_grad(const double* weights, double* weightsb) {
        double w = 1.0/len;
        for(size_t i=0; i<NW; i++) weightsb[i] = 0;
        for(size_t i=0; i<len; i++) NeuralNetworkLoss_b(&images[N*i], weights, weightsb, &label_dists[O*i], w);
    }
};

image_set train;
image_set test;

int main() {
    srand(10);
    train.read("data/train");
    test.read("data/t10k");

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
            printf("train: ");
            double loss = train.avg_loss(x);
            printf("test:  ");
            test.avg_loss(x);
            train.avg_loss_grad(x,grad);
            return loss;
        },
        NULL
    );
    opt_res = nlopt_set_maxeval(opt, 200);
    double obj;
    opt_res = nlopt_optimize(opt, weights.data(), &obj);
    printf("obj: %lg", obj);
    return 0;
}
