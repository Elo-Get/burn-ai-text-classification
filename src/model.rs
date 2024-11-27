use burn::{tensor::backend::Backend};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Relu, Initializer};
use burn::nn::Initializer::Zeros;
use burn::tensor::{Tensor, Int};
pub struct TextClassifier<B: Backend> {
    embedding: Embedding<B>,  // Couche d'embedding
    fc1: Linear<B>,           // Première couche linéaire
    relu: Relu,            // Activation ReLU
    output: Linear<B>,        // Couche de sortie
}


impl<B: Backend> TextClassifier<B> {
    

    /**
     * vocab_size: Taille du vocabulaire => Nombre de mots uniques = taille du vecteur de vocabulaire
     * embedding_dim: Dimension vectorielle de chaque token
     * hidden_dim: Dimension de la couche cachée
     * num_classes: Nombre de classes de sortie => dans notre cas 7 classes car 7 catégories
     * device: Périphérique sur lequel les tenseurs seront alloués
     */


    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let embedding_config = EmbeddingConfig::new(vocab_size, embedding_dim)
            .with_initializer(Zeros);  // Initialisation avec des zéros
        let fc1_config = LinearConfig::new(embedding_dim, hidden_dim)
            .with_initializer(Zeros);  // Initialisation avec des zéros
        let output_config = LinearConfig::new(hidden_dim, num_classes)
            .with_initializer(Zeros);  // Initialisation avec des zéros

        // Initialisation des couches via la méthode `init`
        TextClassifier {
            embedding: embedding_config.init(device),
            fc1: fc1_config.init(device),
            relu: Relu::new(),
            output: output_config.init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        // Passage par la couche d'embedding (entrée de type Int)
        let x = self.embedding.forward(input); // Ici, on s'assure que l'entrée est de type Int

        // Récupérer les dimensions du tenseur après la couche d'Embedding
        let shape = x.shape();  // Récupère l'objet Shape
        let dims = shape.dims;  // Accède aux dimensions du tenseur sous forme de Vec<usize>
        let batch_size = dims[0];
        let hidden_dim = dims[1];

        // Passage par la première couche linéaire
        let x = self.fc1.forward(x);
        
        // Activation ReLU
        let x = self.relu.forward(x);
        
        // Aplatissement de la sortie avant de la passer à la couche de sortie
        let x = x.reshape([batch_size, hidden_dim]); // [batch_size, hidden_dim]
        
        // Passage par la couche de sortie
        self.output.forward(x)
    }


}