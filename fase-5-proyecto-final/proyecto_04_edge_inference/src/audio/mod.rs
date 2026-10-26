//! Pipeline de procesamiento de audio en tiempo real y extracción de espectrogramas Mel.

pub mod postprocessor;
pub mod resampler;
pub mod spectrogram;
pub mod types;
pub mod wav;

pub use postprocessor::AudioPostprocessor;
pub use resampler::AudioResampler;
pub use spectrogram::MelSpectrogramExtractor;
pub use types::{AudioInferenceResult, AudioKeywordPrediction, AudioTiming, SpectrogramConfig};
pub use wav::WavDecoder;
