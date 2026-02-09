"""
Persistence Module for Tensor Risk Engine
==========================================
Handles saving and loading complete analysis sessions using ZIP archives.

File Format:
- ZIP bundle containing:
  - metadata.json: Human-readable configuration and summary
  - inputs.npz: Compressed NumPy arrays (v, u, C, x_grid, H, lambdas)
  - results.npz: Compressed NumPy arrays (all metrics, loss_matrix)
  - gradients.npz: Compressed NumPy arrays (grad_v, grad_C, grad_H, grad_lambdas)
"""

import numpy as np
import json
import zipfile
from io import BytesIO
from datetime import datetime
from typing import Dict, Optional, Tuple
import streamlit as st


def save_analysis(inputs: Dict, results: Dict, gradients: Optional[Dict], 
                 metadata: Dict) -> BytesIO:
    """
    Save complete analysis to ZIP archive.
    
    Parameters
    ----------
    inputs : dict
        Input arrays: v, u, C, x_grid, H, lambdas
    results : dict
        Results from engine: loss_matrix, all metrics
    gradients : dict or None
        Gradient arrays if computed: grad_v, grad_C, grad_H, grad_lambdas
    metadata : dict
        Session metadata: timestamp, dimensions, data_source, etc.
    
    Returns
    -------
    BytesIO
        ZIP file buffer ready for download
    """
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. Save metadata as JSON
        metadata_json = json.dumps(metadata, indent=2, default=str)
        zf.writestr('metadata.json', metadata_json)
        
        # 2. Save inputs as compressed NPZ
        inputs_buffer = BytesIO()
        np.savez_compressed(
            inputs_buffer,
            v=inputs['v'],
            u=inputs['u'],
            C=inputs['C'],
            x_grid=inputs['x_grid'],
            H=inputs['H'],
            lambdas=inputs['lambdas']
        )
        zf.writestr('inputs.npz', inputs_buffer.getvalue())
        
        # 3. Save results as compressed NPZ
        results_buffer = BytesIO()
        np.savez_compressed(
            results_buffer,
            loss_matrix=results['loss_matrix'],
            aal_per_asset=results['aal_per_asset'],
            aal_portfolio=np.array([results['aal_portfolio']]),
            mean_per_event_per_asset=results['mean_per_event_per_asset'],
            variance_per_asset=results['variance_per_asset'],
            std_per_asset=results['std_per_asset'],
            loss_per_event=results['loss_per_event'],
            total_rate=np.array([results['total_rate']])
        )
        zf.writestr('results.npz', results_buffer.getvalue())
        
        # 4. Save gradients if computed
        if gradients is not None:
            gradients_buffer = BytesIO()
            np.savez_compressed(
                gradients_buffer,
                grad_v=gradients['grad_exposure'],
                grad_C=gradients['grad_vulnerability'],
                grad_H=gradients['grad_hazard'],
                grad_lambdas=gradients['grad_lambdas']
            )
            zf.writestr('gradients.npz', gradients_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer


def load_analysis(uploaded_file) -> Dict:
    """
    Load complete analysis from ZIP archive.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object (ZIP format)
    
    Returns
    -------
    dict
        Complete analysis data with keys:
        - inputs: dict with v, u, C, x_grid, H, lambdas
        - results: dict with all metrics and loss_matrix
        - gradients: dict or None if not present
        - metadata: dict with session information
    
    Raises
    ------
    ValueError
        If ZIP structure is invalid or required files are missing
    """
    try:
        # Read ZIP file
        zip_buffer = BytesIO(uploaded_file.read())
        
        with zipfile.ZipFile(zip_buffer, 'r') as zf:
            # 1. Load metadata
            if 'metadata.json' not in zf.namelist():
                raise ValueError("Missing metadata.json in ZIP file")
            
            metadata_str = zf.read('metadata.json').decode('utf-8')
            metadata = json.loads(metadata_str)
            
            # 2. Load inputs
            if 'inputs.npz' not in zf.namelist():
                raise ValueError("Missing inputs.npz in ZIP file")
            
            inputs_buffer = BytesIO(zf.read('inputs.npz'))
            inputs_npz = np.load(inputs_buffer)
            inputs = {
                'v': inputs_npz['v'],
                'u': inputs_npz['u'],
                'C': inputs_npz['C'],
                'x_grid': inputs_npz['x_grid'],
                'H': inputs_npz['H'],
                'lambdas': inputs_npz['lambdas']
            }
            
            # 3. Load results
            if 'results.npz' not in zf.namelist():
                raise ValueError("Missing results.npz in ZIP file")
            
            results_buffer = BytesIO(zf.read('results.npz'))
            results_npz = np.load(results_buffer)
            results = {
                'loss_matrix': results_npz['loss_matrix'],
                'aal_per_asset': results_npz['aal_per_asset'],
                'aal_portfolio': float(results_npz['aal_portfolio'][0]),
                'mean_per_event_per_asset': results_npz['mean_per_event_per_asset'],
                'variance_per_asset': results_npz['variance_per_asset'],
                'std_per_asset': results_npz['std_per_asset'],
                'loss_per_event': results_npz['loss_per_event'],
                'total_rate': float(results_npz['total_rate'][0])
            }
            
            # 4. Load gradients if present
            gradients = None
            if 'gradients.npz' in zf.namelist():
                gradients_buffer = BytesIO(zf.read('gradients.npz'))
                gradients_npz = np.load(gradients_buffer)
                gradients = {
                    'grad_exposure': gradients_npz['grad_v'],
                    'grad_vulnerability': gradients_npz['grad_C'],
                    'grad_hazard': gradients_npz['grad_H'],
                    'grad_lambdas': gradients_npz['grad_lambdas']
                }
        
        return {
            'inputs': inputs,
            'results': results,
            'gradients': gradients,
            'metadata': metadata
        }
    
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file format")
    except KeyError as e:
        raise ValueError(f"Missing required array in NPZ file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading analysis: {str(e)}")


def validate_loaded_data(data: Dict) -> Tuple[bool, str]:
    """
    Validate loaded analysis data for consistency.
    
    Parameters
    ----------
    data : dict
        Loaded data from load_analysis()
    
    Returns
    -------
    valid : bool
        True if data is valid
    message : str
        Error message if invalid, success message if valid
    """
    try:
        inputs = data['inputs']
        results = data['results']
        metadata = data['metadata']
        
        # Check dimensions
        N = inputs['v'].shape[0]
        Q = inputs['H'].shape[1]
        K = inputs['C'].shape[0]
        M = inputs['C'].shape[1]
        
        # Validate shapes
        if inputs['u'].shape[0] != N:
            return False, f"Typology vector shape mismatch: {inputs['u'].shape[0]} != {N}"
        
        if inputs['H'].shape[0] != N:
            return False, f"Hazard matrix rows mismatch: {inputs['H'].shape[0]} != {N}"
        
        if inputs['x_grid'].shape[0] != M:
            return False, f"Intensity grid shape mismatch: {inputs['x_grid'].shape[0]} != {M}"
        
        if inputs['lambdas'].shape[0] != Q:
            return False, f"Lambda vector shape mismatch: {inputs['lambdas'].shape[0]} != {Q}"
        
        if results['loss_matrix'].shape != (N, Q):
            return False, f"Loss matrix shape mismatch: {results['loss_matrix'].shape} != ({N}, {Q})"
        
        # Validate metadata dimensions match
        if metadata['dimensions']['N'] != N:
            return False, "Metadata dimension N doesn't match data"
        
        if metadata['dimensions']['Q'] != Q:
            return False, "Metadata dimension Q doesn't match data"
        
        if metadata['dimensions']['K'] != K:
            return False, "Metadata dimension K doesn't match data"
        
        # Validate gradients if present
        if data['gradients'] is not None:
            grads = data['gradients']
            if grads['grad_exposure'].shape[0] != N:
                return False, "Gradient exposure shape mismatch"
            if grads['grad_vulnerability'].shape != (K, M):
                return False, "Gradient vulnerability shape mismatch"
            if grads['grad_hazard'].shape != (N, Q):
                return False, "Gradient hazard shape mismatch"
            if grads['grad_lambdas'].shape[0] != Q:
                return False, "Gradient lambdas shape mismatch"
        
        return True, f"✓ Valid analysis: {N} assets, {Q} events, {K} typologies"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def create_metadata(inputs: Dict, results: Dict, gradients: Optional[Dict],
                   data_source: str, uploaded_filenames: Optional[Dict],
                   lambda_mode: str, computation_time: float) -> Dict:
    """
    Create metadata dictionary for saving.
    
    Parameters
    ----------
    inputs : dict
        Input arrays
    results : dict
        Results from engine
    gradients : dict or None
        Gradients if computed
    data_source : str
        'synthetic', 'uploaded', or 'loaded'
    uploaded_filenames : dict or None
        Original filenames if uploaded
    lambda_mode : str
        'uniform', 'exponential', or 'custom'
    computation_time : float
        Computation time in seconds
    
    Returns
    -------
    dict
        Metadata dictionary
    """
    N = inputs['v'].shape[0]
    Q = inputs['H'].shape[1]
    K = inputs['C'].shape[0]
    M = inputs['C'].shape[1]
    
    metadata = {
        'version': '1.0',
        'engine': 'TensorialRiskEngine',
        'timestamp': datetime.now().isoformat(),
        'dimensions': {
            'N': int(N),
            'Q': int(Q),
            'K': int(K),
            'M': int(M)
        },
        'data_source': data_source,
        'uploaded_filenames': uploaded_filenames or {},
        'lambda_mode': lambda_mode,
        'gradients_computed': gradients is not None,
        'computation_time_seconds': float(computation_time),
        'total_rate': float(results['total_rate']),
        'aal_portfolio': float(results['aal_portfolio'])
    }
    
    return metadata
