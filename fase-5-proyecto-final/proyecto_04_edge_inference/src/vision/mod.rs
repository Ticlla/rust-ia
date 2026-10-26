//! Pipeline de visión por computadora para inferencia en el borde (Edge).

pub mod postprocessor;
pub mod preprocessor;
pub mod types;

pub use postprocessor::VisionPostprocessor;
pub use preprocessor::VisionPreprocessor;
pub use types::{LetterboxInfo, VisionClassPrediction, VisionInferenceResult, VisionTiming};
