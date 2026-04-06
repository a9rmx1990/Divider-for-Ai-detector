use crate::error::{Result, ScannerError};
use ndarray::{Array2, Array3};
use num_complex::Complex;
use std::f32::consts::PI;

/// Frequency analyzer using 2D Discrete Cosine Transform (DCT)
pub struct FrequencyAnalyzer {
    dct_cache: Option<DctCache>,
}

/// Pre-computed DCT cosine values for faster computation
struct DctCache {
    size: usize,
    cos_table: Vec<Vec<f32>>,
}

impl DctCache {
    fn new(size: usize) -> Self {
        let mut cos_table = vec![vec![0.0; size]; size];
        for u in 0..size {
            for x in 0..size {
                let value = ((2.0 * x as f32 + 1.0) * u as f32 * PI) / (2.0 * size as f32);
                cos_table[u][x] = value.cos();
            }
        }
        Self { size, cos_table }
    }

    fn get(&self, u: usize, x: usize) -> f32 {
        self.cos_table[u][x]
    }
}

impl FrequencyAnalyzer {
    /// Create a new frequency analyzer
    pub fn new() -> Self {
        Self { dct_cache: None }
    }

    /// Create with DCT caching for optimal performance
    pub fn with_cache(patch_size: usize) -> Self {
        Self {
            dct_cache: Some(DctCache::new(patch_size)),
        }
    }

    /// Compute 2D DCT for a single channel (grayscale)
    /// 
    /// # Arguments
    /// * `input` - 2D array [height, width]
    /// 
    /// # Returns
    /// 2D DCT coefficients
    pub fn compute_dct_2d(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (height, width) = input.dim();

        if height != width {
            return Err(ScannerError::DctError(
                "DCT requires square patches".to_string(),
            ));
        }

        let size = height;
        let mut output = Array2::<f32>::zeros((size, size));

        // Compute DCT using separable property: 1D row-wise, then 1D column-wise
        // First: DCT on rows
        let mut temp = Array2::<f32>::zeros((size, size));
        for i in 0..size {
            let row = input.row(i).to_owned();
            let dct_row = self.compute_dct_1d(&row.to_vec(), size)?;
            for j in 0..size {
                temp[[i, j]] = dct_row[j];
            }
        }

        // Second: DCT on columns
        for j in 0..size {
            let col = temp.column(j).to_owned().to_vec();
            let dct_col = self.compute_dct_1d(&col, size)?;
            for i in 0..size {
                output[[i, j]] = dct_col[i];
            }
        }

        Ok(output)
    }

    /// Compute 1D DCT
    fn compute_dct_1d(&self, input: &[f32], n: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0; n];

        for u in 0..n {
            let mut sum = 0.0;
            for x in 0..n {
                let cos_val = if let Some(ref cache) = self.dct_cache {
                    cache.get(u, x)
                } else {
                    (((2.0 * x as f32 + 1.0) * u as f32 * PI) / (2.0 * n as f32)).cos()
                };
                sum += input[x] * cos_val;
            }

            let alpha = if u == 0 { 1.0 / (n as f32).sqrt() } else { (2.0 / n as f32).sqrt() };
            output[u] = alpha * sum;
        }

        Ok(output)
    }

    /// Extract log-magnitude spectrum (critical for forensics)
    /// 
    /// Returns log(1 + |DCT|) to identify AI fingerprints in high frequencies
    pub fn get_log_magnitude_spectrum(&self, dct: &Array2<f32>) -> Result<Array2<f32>> {
        let log_spectrum = dct.mapv(|x| (1.0 + x.abs()).ln());
        Ok(log_spectrum)
    }

    /// Extract phase spectrum from DCT
    pub fn get_phase_spectrum(&self, dct: &Array2<f32>) -> Result<Array2<f32>> {
        let phase = dct.mapv(|x| x.atan());
        Ok(phase)
    }

    /// Compute frequency band statistics
    ///
    /// DCT layout: DC component is at [0, 0].  Frequency increases with
    /// distance from the origin (top-left corner).
    ///   - Low  freq: distance from (0,0) < N/4
    ///   - Mid  freq: distance from (0,0) in [N/4, N/2)
    ///   - High freq: distance from (0,0) >= N/2
    pub fn get_frequency_bands(&self, dct: &Array2<f32>) -> Result<FrequencyBandStats> {
        let (height, width) = dct.dim();
        let n = height.max(width) as f32;

        // DC is at [0, 0] in DCT
        let dc_component = dct[[0, 0]];

        let lf_bound = n / 4.0;   // 0 .. N/4
        let mf_bound = n / 2.0;   // N/4 .. N/2
        // hf = everything beyond N/2

        let lf_bound_sq = lf_bound * lf_bound;
        let mf_bound_sq = mf_bound * mf_bound;

        let mut lf_energy = 0.0_f32;
        let mut lf_count = 0_u32;
        let mut mf_energy = 0.0_f32;
        let mut mf_count = 0_u32;
        let mut hf_energy = 0.0_f32;
        let mut hf_count = 0_u32;

        // Collect log-magnitude and magnitude sum for spectral flatness
        let mut sum_mag = 0.0_f32;
        let mut sum_log_mag = 0.0_f32;
        let mut count_mag = 0_f32;

        for i in 0..height {
            for j in 0..width {
                let dist_sq = (i * i + j * j) as f32;
                let mag = dct[[i, j]].abs();

                // Skip pure DC for band energy (it dominates and distorts ratios)
                if i == 0 && j == 0 {
                    let val = mag.max(1e-9);
                    sum_mag += val;
                    sum_log_mag += val.ln();
                    count_mag += 1.0;
                    continue;
                }

                if dist_sq < lf_bound_sq {
                    lf_energy += mag;
                    lf_count += 1;
                } else if dist_sq < mf_bound_sq {
                    mf_energy += mag;
                    mf_count += 1;
                } else {
                    hf_energy += mag;
                    hf_count += 1;
                }

                let val = mag.max(1e-9);
                sum_mag += val;
                sum_log_mag += val.ln();
                count_mag += 1.0;
            }
        }

        // Normalise to mean energy per coefficient
        if lf_count > 0 { lf_energy /= lf_count as f32; }
        if mf_count > 0 { mf_energy /= mf_count as f32; }
        if hf_count > 0 { hf_energy /= hf_count as f32; }

        let hf_lf_ratio = if lf_energy > 1e-9 { hf_energy / lf_energy } else { 0.0 };

        // Spectral flatness: geometric_mean / arithmetic_mean of magnitudes.
        // Flatness ≈ 1.0 (White noise), ≈ 0.0 (Tonal/Photo).
        let spectral_flatness = if count_mag > 0.0 {
            let arith_mean = sum_mag / count_mag;
            let geo_mean = (sum_log_mag / count_mag).exp();
            if arith_mean > 1e-9 {
                geo_mean / arith_mean
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(FrequencyBandStats {
            dc_component,
            low_freq_energy: lf_energy,
            mid_freq_energy: mf_energy,
            high_freq_energy: hf_energy,
            hf_lf_ratio,
            spectral_flatness,
        })
    }

    /// Multi-channel DCT (grayscale → 1 channel, RGB → 3 channels)
    pub fn compute_dct_multichannel(&self, patch: &Array3<f32>) -> Result<Vec<Array2<f32>>> {
        let (height, width, channels) = patch.dim();

        if height != width {
            return Err(ScannerError::DctError(
                "DCT requires square patches".to_string(),
            ));
        }

        let mut dct_channels = Vec::new();

        for c in 0..channels {
            let channel_data = patch.slice(ndarray::s![.., .., c]);
            let channel_2d = channel_data.into_owned();
            let dct = self.compute_dct_2d(&channel_2d)?;
            dct_channels.push(dct);
        }

        Ok(dct_channels)
    }

    /// Detect high-frequency artifacts (AI fingerprints)
    ///
    /// Uses statistical Z-score peak detection. Natural images have smooth random 
    /// decay in high frequencies (HF). AI models leaving behind periodic checkerboard 
    /// aliasing will manifest as severe magnitude spikes (peaks) in the HF band.
    pub fn detect_hf_artifacts(&self, dct: &Array2<f32>, _threshold_obsolete: f32) -> Result<ArtifactDetection> {
        let (height, width) = dct.dim();
        let total_coeffs = height * width;
        let n = height.max(width) as f32;

        let mut hf_coeffs = Vec::new();

        // High frequency = coefficients beyond N/2 from origin
        let hf_bound_sq = (n / 2.0) * (n / 2.0);

        for i in 0..height {
            for j in 0..width {
                let dist_sq = (i * i + j * j) as f32;

                if dist_sq >= hf_bound_sq {
                    hf_coeffs.push(dct[[i, j]].abs());
                }
            }
        }

        let total_hf = hf_coeffs.len();
        if total_hf == 0 {
            return Ok(ArtifactDetection {
                anomaly_detected: false,
                anomaly_score: 0.0,
                artifact_magnitude: 0.0,
                artifact_count: 0,
            });
        }

        // 1. Calculate Mean
        let sum: f32 = hf_coeffs.iter().sum();
        let mean = sum / total_hf as f32;

        // 2. Calculate Standard Deviation
        let variance: f32 = hf_coeffs.iter().map(|&x| {
            let diff = x - mean;
            diff * diff
        }).sum::<f32>() / total_hf as f32;
        let std_dev = variance.sqrt();

        // 3. Peak Detection (Z-Score > 5.0)
        // A 5-sigma event in natural noise is extremely rare (~1 in 3.5 million).
        // Since a 256x256 patch has ~32,000 HF coefficients, any cluster of 
        // 5-sigma spikes strongly indicates an unnatural periodic grid pattern.
        let z_threshold = mean + (5.0 * std_dev);

        let mut hf_count = 0_usize;
        let mut hf_magnitude_sum = 0.0_f32;

        for &mag in &hf_coeffs {
            if mag > z_threshold {
                hf_count += 1;
                hf_magnitude_sum += mag;
            }
        }

        // Score based on number of unnatural peaks found in the HF band
        let hf_anomaly_score = hf_count as f32 / total_hf as f32;

        Ok(ArtifactDetection {
            // Require at least 4 spikes to confidently declare a "checkerboard grid", 
            // mitigating a random singular noise fluctuation.
            anomaly_detected: hf_count >= 4,
            anomaly_score: hf_anomaly_score,
            artifact_magnitude: if hf_count > 0 { hf_magnitude_sum / hf_count as f32 } else { 0.0 },
            artifact_count: hf_count,
        })
    }
}

/// Frequency band energy statistics
#[derive(Clone, Debug)]
pub struct FrequencyBandStats {
    pub dc_component: f32,
    pub low_freq_energy: f32,
    pub mid_freq_energy: f32,
    pub high_freq_energy: f32,
    pub hf_lf_ratio: f32,
    pub spectral_flatness: f32,
}

/// Artifact detection results
#[derive(Clone, Debug)]
pub struct ArtifactDetection {
    pub anomaly_detected: bool,
    pub anomaly_score: f32,
    pub artifact_magnitude: f32,
    pub artifact_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_1d() {
        let analyzer = FrequencyAnalyzer::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = analyzer.compute_dct_1d(&input, 4).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_dct_2d() {
        let analyzer = FrequencyAnalyzer::with_cache(256);
        let input = Array2::<f32>::ones((256, 256));
        let result = analyzer.compute_dct_2d(&input).unwrap();
        assert_eq!(result.dim(), (256, 256));
    }

    #[test]
    fn test_log_magnitude() {
        let analyzer = FrequencyAnalyzer::new();
        let dct = Array2::<f32>::from_elem((4, 4), 1.0);
        let log_spec = analyzer.get_log_magnitude_spectrum(&dct).unwrap();
        assert!(log_spec.iter().all(|&x| x > 0.0));
    }
}
