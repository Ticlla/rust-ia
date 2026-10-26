//! Tipos y estructuras del motor de inferencia ONNX Runtime.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelKind {
    Vision,
    Audio,
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vision => write!(f, "Vision"),
            Self::Audio => write!(f, "Audio"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum EngineError {
    SessionInit(String),
    InferenceFailed(String),
    PoolExhausted,
    ModelNotFound(String),
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionInit(msg) => write!(f, "Error inicializando sesión ONNX: {msg}"),
            Self::InferenceFailed(msg) => write!(f, "Fallo durante inferencia ONNX: {msg}"),
            Self::PoolExhausted => write!(f, "Pool de sesiones ONNX agotado (timeout)"),
            Self::ModelNotFound(path) => write!(f, "Modelo ONNX no encontrado en: {path}"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Discrepancia de tensor: esperado {expected} elementos, recibidos {got}")
            }
        }
    }
}

impl std::error::Error for EngineError {}
