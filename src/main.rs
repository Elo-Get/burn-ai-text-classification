mod dataset;
mod vocabulary;
mod preprocess;
mod model;


use std::path::Path;
use burn::{ backend::{wgpu::{init_sync, Metal, WgpuDevice}, Wgpu}, nn::loss::CrossEntropyLoss, optim::{decay::WeightDecayConfig, AdamW, AdamWConfig, Optimizer}, tensor::{Int, Tensor}};
use dataset::Dataset;
use model::TextClassifier;

type TheBackend = Wgpu<f32, i32>;

#[derive(Debug)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

fn main() {
    let model_path = "model.json";
    let mut model = if Path::new(model_path).exists() {
        
        // Charger le modèle
        // TO DO

        unimplemented!()



    } else {

        // Charger et prétraiter les données
        let dataset = Dataset::from_json("dataset.json").expect("Erreur de chargement des données");
        let mut vocab = vocabulary::Vocabulary::new();
        let max_len = 10;
        let preprocessed = preprocess::preprocess_dataset(dataset, &mut vocab, max_len);

        // Création du model
        let vocab_size = vocab.size();      // Taille du vocabulaire
        let embedding_dim = 128;       // Dimension de l'embedding (exemple)
        let hidden_dim = 64;           // Dimension de la couche cachée (exemple)
        let num_classes = 7;           // Nombre de classes (exemple)

        // Initialisation du backend
        let device = WgpuDevice::default(); // Par défaut, détecte le GPU ou revient au CPU
        let backend = init_sync::<Metal>(&device, Default::default()); // Utilise Metal pour macOS
        
        type Model = TextClassifier<TheBackend>;
        
        let model: TextClassifier<TheBackend> = TextClassifier::new(vocab_size, embedding_dim, hidden_dim, num_classes, &device);


        // Fonction de perte et optimiseur
        let criterion: CrossEntropyLoss<TheBackend> = CrossEntropyLoss::new(None, &device);
        let optimizer = burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))).init();
        
        // Entraînement
        for epoch in 0..1000 {
            for example in &preprocessed {
                let input_tensor: Tensor<TheBackend, 2, Int> = Tensor::from(example.tokens.clone());  // Exemple de création de tensor
                let target_tensor: Tensor<TheBackend, 2, Int> = Tensor::from(vec![example.label]);
        
                // Zero gradients
                optimizer.zero_grand();
        
                // Forward pass
                let output = model.forward(input_tensor);
        
                // Calcul de la perte
                let loss: Tensor<burn_fusion::Fusion<burn::backend::wgpu::JitBackend<burn::backend::wgpu::WgpuRuntime, f32, i32>>, 1> = criterion.forward(&output, &target_tensor);
        
                // Backward pass
                loss.backward();
        
                // Step de l'optimiseur
                optimizer.step();

                let lr = 0.001;
                let momentum = 0.9;
                let weight_decay = 0.0;
                optimizer.step(lr, momentum, weight_decay);
            }
        }

        model

    };

    println!("Le modèle est prêt pour les prédictions.");


    // Prédiction
    // TO DO

}


