#include <cmath>
#include <cstddef>
#include <iostream> 
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>


#include "inculdes/math_operations.h"

struct Image {
    int label;
    std::vector<double> pixels;
    std::vector<double> targetVector;
};

double sigmoid(double x){
    return 1.0/(1.0 + std::exp(-x));
}

double sigmoid_prime(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

std::vector<double> createTargetVector(const int &label){
    if(label > 9) std::cerr << "createTargetVector: label kann nicht grösser 9 sein" << std::endl;
    std::vector<double> targetVector(10);
    for(int i = 0; i < 10; ++i){
        if(i == label){
            targetVector[i] = 1.0;
        } else{
            targetVector[i] = 0.0;
        }
    }
    return targetVector;
}

/*
l = Index für die Schichtübergänge (von Schicht l zu Schicht l+1)

j = Index eines Neurons in der nächsten Schicht (also Schicht l+1)

k = Index eines Neurons in der aktuellen Schicht (also Schicht l)
*/

struct NeuralNetwork {
    std::vector<int> architecture; // 1. Anzahl Inputs 2. Anzahl Neuronen pro Spalte Hidden-Layer 3. Anzahl Outputs
    std::vector<std::vector<double>> biases; // biases[layer][neuron] = Bias des Neurons in dieser Schicht
    std::vector<std::vector<std::vector<double>>> weights; // weights[layer][target_neuron][source_neuron] = Gewicht der Verbindung
    std::vector<std::vector<double>> aPerLayer; // wird später bei der backpropagation gebraucht. Spichert alle current_a
    std::vector<std::vector<double>> zVectorPerLayer; // wird später bei der backpropagation gebraucht. Spichert alle zVektoren also jeden wert
    std::vector<std::vector<std::vector<double>>> nabla_w;
    std::vector<std::vector<double>> nabla_b;

    NeuralNetwork(std::vector<int> architecture){
        this->architecture = architecture;
        for(size_t l = 1; l < architecture.size(); ++l){ // l = Anzahl Layer 
            biases.push_back(random_vector(architecture[l]));
            weights.push_back(random_matrix(architecture[l], architecture[l-1]));
        }
    }

    std::vector<double> feedforward(const std::vector<double> &imagePixels){    
        if(imagePixels.size() != architecture[0]) std::cerr << "Quantity of Pixels does not match architecture" << std::endl;
        
        aPerLayer.clear();  //setzt vor jedem durchlauf die vektoren die für die backprop benötigt werden zurück
        zVectorPerLayer.clear();


        std::vector<double> current_a = imagePixels; // erster input also die pixel eines bildes
        aPerLayer.push_back(current_a);
        for(size_t l = 0; l < biases.size(); ++l){   // iteriert über die layer

            //es wird das z für jedes neuron berechnet
            std::vector<double> zVector = vectorAddition(matrixVectorMultiplication
                (weights[l], current_a), biases[l]); 
            //speichert alle z für später in der Backpropagation
            zVectorPerLayer.push_back(zVector);
            //führt für jedes z die Aktivierungsfunktion (hier Sigmoid) aus
            std::vector<double> tmp;
            for(double z : zVector){
                tmp.push_back(sigmoid(z));
            }
            current_a = tmp;
            //alle aktivierungen dieses layers welche zum input des nächsten layer werden, weden für später zwischen gespeichert
            aPerLayer.push_back(current_a);
        }
        return current_a; // output am ende mit welchem der Loss berechnet wird
    }

    std::vector<double> createPrimeSigmoidVector(std::vector<double> zVector){
        std::vector<double> prime_sigmoid_vector(zVector.size());
        for(size_t i = 0; i < prime_sigmoid_vector.size(); ++i){
            prime_sigmoid_vector[i] = sigmoid_prime(zVector[i]);
        }
        return prime_sigmoid_vector;
    }

    double backpropagation(const std::vector<double> &utterTrashOutput, const std::vector<double> &targetVector){
        if(utterTrashOutput.size() != targetVector.size()) std::cerr << "backpropagation: size missmatch" << std::endl;
        
        // ------------- Loss und Cost berechnen --------------------
        std::vector<double> loss(utterTrashOutput.size());
        for(size_t i = 0; i < utterTrashOutput.size(); ++i){
            loss[i] = std::pow(utterTrashOutput[i] - targetVector[i], 2);
        }
        //cost ist der Durchschnitt von loss
        double cost = averageOfVector(loss);

        // ------------------ Backpropagation -----------------------
        std::vector<std::vector<std::vector<double>>> nabla_wC_per_layer(zVectorPerLayer.size()); 
            // ------------------- Output-Layer -------------------------
        size_t index_output = zVectorPerLayer.size() - 1;
                // Berechnung delta der Ausgabeschicht
        std::vector<std::vector<double>> delta_per_layer(zVectorPerLayer.size());
        std::vector<double> prime_sigmoid_vector = createPrimeSigmoidVector(zVectorPerLayer[index_output]);
        
        std::vector<double> output_delta = hadamardProduct(subtractVectors(utterTrashOutput, targetVector), prime_sigmoid_vector);
        delta_per_layer[index_output] = output_delta;
                // Berechnung der Gradienten der Ausgabeschicht
        std::vector<std::vector<double>> nabla_wC = outerProduct(delta_per_layer[index_output] , aPerLayer[index_output-1]);
        nabla_wC_per_layer[index_output] = nabla_wC;
            // ------------------- Hidden-Layers ------------------------
        for(int l = index_output-1; l >= 0; --l){
            //Berechnung des neuen delta. Dieberechnung von delta in den Hidden-Layers ist anders
            const std::vector<double> &prev_delta = delta_per_layer[l+1];
            std::vector<double> current_delta = hadamardProduct(matrixVectorMultiplication
                (transposeMatrix(weights[l]), prev_delta), createPrimeSigmoidVector(zVectorPerLayer[l]));
        
            delta_per_layer[l] = current_delta;

            //Berechnung des Gradienten dieses Layers
            nabla_wC = outerProduct(delta_per_layer[l] , aPerLayer[l-1]);
            nabla_wC_per_layer[l] = nabla_wC;
        }

        nabla_w = nabla_wC_per_layer;
        nabla_b = delta_per_layer;
        return cost;
    }

    void updateWeightsAndBiases(double learning_rate){
        for(size_t l = 0; l < zVectorPerLayer.size(); ++l){
            // Weights
            std::vector<std::vector<double>> tmpMatrix = matrixScalarMultiplication(nabla_w[l], learning_rate);
            weights[l] = matrixSubtraction(weights[l], tmpMatrix); 

            // Biases
            std::vector<double> tmpVector = vectorScalarMultiplication(nabla_b[l], learning_rate);
            biases[l] = subtractVectors(biases[l], tmpVector);
        }

    }
};

std::vector<Image> readData(std::string path){
    std::ifstream input(path);
        if (!input.is_open()) {
        std::cerr << "Fehler: Datei " << path << " konnte nicht geöffnet werden. Bitte Pfad überprüfen!" << std::endl;
    }
    std::string line;
    std::vector<Image> images;
    short c = 0;
    while(getline(input, line)){
        if(c < 1) {                             // avoid header line
            c += 1;
            continue;
        }
        std::stringstream ss(line);
        std::string segment;
        short count = 0;
        Image image;
        while(std::getline(ss, segment,',')){
            if(count < 1){
                image.label = std::stoi(segment);
                count += 1;
            } else {
                image.pixels.push_back(std::stod(segment));
            }
        }
        image.targetVector = createTargetVector(image.label);
        images.push_back(image);
    }
    return images;
}

int main(){
    std::vector<Image> images = readData("data/mnist_train.csv");
    /*std::cout << "First Label: " << images[0].label << std::endl;
    std::cout << "Last Label: " << images[images.size()-1].label << std::endl;
    */



    return 0;
}