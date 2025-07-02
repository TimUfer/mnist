## How to use

1. clone repository 
    
    ```jsx
    git clone git@github.com:TimUfer/mnist.git
    ```
    
2. add mnist csv files to data directory 
3. cd to mnist and compile main.cpp
    
    ```jsx
    g++ main.cpp includes/math_operations.cpp -o main
    ```
    
4. execute main
    
    ```jsx
    ./main
    ```
    
---

## Benutzte Formeln für das Projekt

$$
\nabla_W C^{(L)} = \delta^{(L)} (a^{(L-1)})^T
$$

Sigmoid-Funktion:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Ableitung der Sigmoid-Funktion:

$$
\sigma'(z) = \sigma(z) (1 - \sigma(z))
$$

Berechnung delta Ausgabe-Layer:

```
std::vector<double> output_delta = hadamardProduct(subtractVectors(utterTrashOutput, targetVector), prime_sigmoid_vector);
```

$$
\delta^L = (a^L - y) \odot \sigma'(z^L)
$$

Berechnung delta in den Hidden-Layers:

```
std::vector<double> current_delta = hadamardProduct(matrixVectorMultiplication(transposeMatrix(weights[l]), prev_delta), createPrimeSigmoidVector(zVectorPerLayer[l]));
```

$$
\delta^l = ((W^{l})^T \delta^{l+1}) \odot \sigma'(z^l)
$$

Gradient für die Biases:

$$
\frac{\partial C}{\partial b^l} = \delta^l
$$

Gradient für die Gewichte:

// Für die Ausgabeschicht (l=L):

```
std::vector<std::vector<double>> nabla_wC = outerProduct(delta_per_layer[index_output], aPerLayer[index_output]);
```

// Für versteckte Schichten (l):

```
nabla_wC = outerProduct(delta_per_layer[l], aPerLayer[l]);
```

$$
\frac{\partial C}{\partial W^l} = \delta^l (a^{l-1})^T
$$
