//! Preprocesador de imágenes de alta velocidad con letterboxing, normalización y conversión NCHW.

use crate::vision::types::LetterboxInfo;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb, RgbImage};
use std::io::Cursor;

/// Parámetros estándar de normalización ImageNet
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct VisionPreprocessor {
    pub target_w: u32,
    pub target_h: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for VisionPreprocessor {
    fn default() -> Self {
        Self {
            target_w: 224,
            target_h: 224,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }
}

impl VisionPreprocessor {
    pub fn new(target_w: u32, target_h: u32) -> Self {
        Self {
            target_w,
            target_h,
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
        }
    }

    /// Decodifica una imagen desde un buffer en memoria (PNG, JPEG, WebP)
    pub fn decode_bytes(&self, bytes: &[u8]) -> Result<DynamicImage, String> {
        image::load_from_memory(bytes).map_err(|e| format!("Fallo decodificando imagen: {e}"))
    }

    /// Decodifica una imagen desde una cadena Base64
    pub fn decode_base64(&self, b64_str: &str) -> Result<DynamicImage, String> {
        use base64::Engine;
        let clean_b64 = if let Some(idx) = b64_str.find(";base64,") {
            &b64_str[idx + 8..]
        } else {
            b64_str
        };

        let decoded_bytes = base64::engine::general_purpose::STANDARD
            .decode(clean_b64.trim())
            .map_err(|e| format!("Fallo decodificando Base64: {e}"))?;

        self.decode_bytes(&decoded_bytes)
    }

    /// Redimensiona manteniendo la relación de aspecto (Letterbox) rellenando bordes
    pub fn letterbox(&self, img: &DynamicImage) -> (RgbImage, LetterboxInfo) {
        let (orig_w, orig_h) = img.dimensions();
        let scale =
            (self.target_w as f32 / orig_w as f32).min(self.target_h as f32 / orig_h as f32);

        let new_w = ((orig_w as f32 * scale).round() as u32)
            .min(self.target_w)
            .max(1);
        let new_h = ((orig_h as f32 * scale).round() as u32)
            .min(self.target_h)
            .max(1);

        let pad_x = (self.target_w - new_w) / 2;
        let pad_y = (self.target_h - new_h) / 2;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb_resized = resized.to_rgb8();

        // Canvas gris neutro (114, 114, 114 estándar para visión edge / YOLO)
        let mut canvas = RgbImage::from_pixel(self.target_w, self.target_h, Rgb([114, 114, 114]));

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = rgb_resized.get_pixel(x, y);
                canvas.put_pixel(pad_x + x, pad_y + y, *pixel);
            }
        }

        let info = LetterboxInfo {
            scale,
            pad_x,
            pad_y,
            new_w,
            new_h,
        };

        (canvas, info)
    }

    /// Convierte imagen RGB HWC a formato plano NCHW normalizado con ImageNet mean/std
    pub fn to_nchw_tensor(&self, canvas: &RgbImage) -> Vec<f32> {
        let w = self.target_w as usize;
        let h = self.target_h as usize;
        let total_pixels = w * h;

        // Prealocación contigua exacta para 3 canales [C, H, W]
        let mut tensor = vec![0.0f32; 3 * total_pixels];

        let (r_slice, rest) = tensor.split_at_mut(total_pixels);
        let (g_slice, b_slice) = rest.split_at_mut(total_pixels);

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let pixel = canvas.get_pixel(x as u32, y as u32);

                // Normalización zero-copy por canal
                r_slice[idx] = (pixel[0] as f32 / 255.0 - self.mean[0]) / self.std[0];
                g_slice[idx] = (pixel[1] as f32 / 255.0 - self.mean[1]) / self.std[1];
                b_slice[idx] = (pixel[2] as f32 / 255.0 - self.mean[2]) / self.std[2];
            }
        }

        tensor
    }

    /// Pipeline completo: bytes -> decodificación -> letterbox -> NCHW tensor
    pub fn process_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<(Vec<f32>, (u32, u32), LetterboxInfo), String> {
        let img = self.decode_bytes(bytes)?;
        let original_dims = img.dimensions();
        let (canvas, letterbox_info) = self.letterbox(&img);
        let tensor = self.to_nchw_tensor(&canvas);

        Ok((tensor, original_dims, letterbox_info))
    }

    /// Generador auxiliar de imágenes sintéticas PNG para pruebas unitarias y benchmarks
    pub fn generate_synthetic_png(width: u32, height: u32) -> Vec<u8> {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = (((x + y) * 128) / (width + height)) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, ImageFormat::Png).unwrap();
        buffer.into_inner()
    }
}
