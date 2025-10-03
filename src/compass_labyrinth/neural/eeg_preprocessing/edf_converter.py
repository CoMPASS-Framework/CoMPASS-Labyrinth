"""
EDF to NumPy Converter Module

This module provides functionality to convert EDF (European Data Format) files
to NumPy .npy format for easier processing and analysis.

Author: Patrick Honma
Date: 06-20-2025
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pyedflib
import mne

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_edf_to_npy(edf_file_path: Union[str, Path], 
                      output_dir: Optional[Union[str, Path]] = None,
                      overwrite: bool = False) -> str:
    """
    Convert a single EDF file to NumPy .npy format.
    
    Parameters:
    -----------
    edf_file_path : str or Path
        Path to the input EDF file
    output_dir : str or Path, optional
        Directory to save the .npy file. If None, saves in same directory as input file
    overwrite : bool, default False
        Whether to overwrite existing .npy files
        
    Returns:
    --------
    str
        Path to the saved .npy file
        
    Raises:
    -------
    FileNotFoundError
        If the input EDF file doesn't exist
    ValueError
        If the EDF file is corrupted or unreadable
    """
    edf_file_path = Path(edf_file_path)
    
    # Validate input file
    if not edf_file_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    
    if not edf_file_path.suffix.lower() == '.edf':
        raise ValueError(f"File must have .edf extension: {edf_file_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = edf_file_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output path
    output_path = output_dir / f"{edf_file_path.stem}.npy"
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        logger.warning(f"Output file already exists: {output_path}. Use overwrite=True to replace.")
        return str(output_path)
    
    try:
        # Open and read EDF file
        logger.info(f"Processing: {edf_file_path}")
        
        with pyedflib.EdfReader(str(edf_file_path)) as edf_reader:
            # Get file information
            num_signals = edf_reader.signals_in_file
            header = edf_reader.getHeader()
            
            # Create data dictionary
            edf_data = {
                'header': header,
                'signals': {},
                'signal_headers': {}  # Store individual signal metadata
            }
            
            # Read all signals
            for i in range(num_signals):
                signal_label = edf_reader.getLabel(i)
                signal_data = edf_reader.readSignal(i)
                signal_header = edf_reader.getSignalHeader(i)
                
                edf_data['signals'][signal_label] = signal_data
                edf_data['signal_headers'][signal_label] = signal_header
                
                logger.debug(f"Read signal: {signal_label} ({len(signal_data)} samples)")
        
        # Save as .npy file
        np.save(output_path, edf_data)
        logger.info(f"Saved: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error processing {edf_file_path}: {str(e)}")
        raise ValueError(f"Failed to process EDF file {edf_file_path}: {str(e)}")


def batch_convert_edf_to_npy(input_dir: Union[str, Path],
                           output_dir: Optional[Union[str, Path]] = None,
                           overwrite: bool = False,
                           file_pattern: str = "*.edf") -> List[str]:
    """
    Convert all EDF files in a directory to NumPy .npy format.
    
    Parameters:
    -----------
    input_dir : str or Path
        Directory containing EDF files
    output_dir : str or Path, optional
        Directory to save .npy files. If None, saves in input_dir
    overwrite : bool, default False
        Whether to overwrite existing .npy files
    file_pattern : str, default "*.edf"
        Pattern to match EDF files (e.g., "*.edf", "subject_*.edf")
        
    Returns:
    --------
    List[str]
        List of paths to converted .npy files
        
    Raises:
    -------
    FileNotFoundError
        If input directory doesn't exist
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all EDF files
    edf_files = list(input_dir.glob(file_pattern))
    
    if not edf_files:
        logger.warning(f"No EDF files found in {input_dir} matching pattern '{file_pattern}'")
        return []
    
    logger.info(f"Found {len(edf_files)} EDF files to convert")
    
    converted_files = []
    failed_files = []
    
    for edf_file in edf_files:
        try:
            output_path = convert_edf_to_npy(edf_file, output_dir, overwrite)
            converted_files.append(output_path)
        except Exception as e:
            logger.error(f"Failed to convert {edf_file}: {str(e)}")
            failed_files.append(str(edf_file))
    
    # Summary
    logger.info(f"Conversion complete:")
    logger.info(f"  Successfully converted: {len(converted_files)} files")
    if failed_files:
        logger.warning(f"  Failed to convert: {len(failed_files)} files")
        for failed_file in failed_files:
            logger.warning(f"    - {failed_file}")
    
    return converted_files


def load_converted_edf(npy_file_path: Union[str, Path]) -> Dict:
    """
    Load a converted EDF file from .npy format.
    
    Parameters:
    -----------
    npy_file_path : str or Path
        Path to the .npy file
        
    Returns:
    --------
    dict
        Dictionary containing 'header', 'signals', and 'signal_headers'
    """
    npy_file_path = Path(npy_file_path)
    
    if not npy_file_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {npy_file_path}")
    
    try:
        data = np.load(npy_file_path, allow_pickle=True).item()
        logger.info(f"Loaded EDF data from: {npy_file_path}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load .npy file {npy_file_path}: {str(e)}")


def get_edf_info(edf_file_path: Union[str, Path]) -> Dict:
    """
    Get basic information about an EDF file without full conversion.
    
    Parameters:
    -----------
    edf_file_path : str or Path
        Path to the EDF file
        
    Returns:
    --------
    dict
        Dictionary with basic file information
    """
    edf_file_path = Path(edf_file_path)
    
    if not edf_file_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    
    try:
        with pyedflib.EdfReader(str(edf_file_path)) as edf_reader:
            header = edf_reader.getHeader()
            signals_info = []
            
            for i in range(edf_reader.signals_in_file):
                signal_info = {
                    'label': edf_reader.getLabel(i),
                    'sample_rate': edf_reader.getSampleFrequency(i),
                    'samples': edf_reader.getNSamples()[i],
                    'physical_dimension': edf_reader.getPhysicalDimension(i)
                }
                signals_info.append(signal_info)
            
            return {
                'filename': edf_file_path.name,
                'num_signals': edf_reader.signals_in_file,
                'duration_seconds': header.get('duration', 'Unknown'),
                'start_date': header.get('startdate', 'Unknown'),
                'signals': signals_info
            }
            
    except Exception as e:
        raise ValueError(f"Failed to read EDF file info: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert EDF files to NumPy format")
    parser.add_argument("input_path", help="Path to EDF file or directory containing EDF files")
    parser.add_argument("-o", "--output", help="Output directory (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--pattern", default="*.edf", help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single file conversion
        convert_edf_to_npy(input_path, args.output, args.overwrite)
    elif input_path.is_dir():
        # Batch conversion
        batch_convert_edf_to_npy(input_path, args.output, args.overwrite, args.pattern)
    else:
        print(f"Error: {input_path} is not a valid file or directory")