import argparse

import h5py

def process_buffer_memory_efficient(input_path, output_path, chunk_size=10000):
    """
    Process the buffer.hdf5 file efficiently by processing one episode at a time.
    
    Args:
        input_path (str): Path to input buffer.hdf5
        output_path (str): Path to save processed output
        chunk_size (int): Number of transitions to process before writing to disk
    """
    with h5py.File(input_path, 'r') as f_in:
        episode_keys = [key for key in f_in.keys() if key.startswith('ep_')]
        
        # First pass: count total number of transitions
        total_transitions = 0
        for ep_key in episode_keys:
            ep = f_in[ep_key]
            total_transitions += len(ep['action']) - 1  # -1 because we need next state
        
        # Get shapes and dtypes
        sample_ep = f_in[episode_keys[0]]
        obs_shape = sample_ep['observation'].shape[1:]
        action_shape = sample_ep['action'].shape[1:]
        qpos_shape = sample_ep['qpos'].shape[1:]
        qvel_shape = sample_ep['qvel'].shape[1:]
        
        # Create output file with chunked storage
        with h5py.File(output_path, 'w') as f_out:
            # Create datasets with chunking and compression
            f_out.create_dataset('action', 
                               shape=(total_transitions, *action_shape),
                               chunks=(chunk_size, *action_shape),
                               compression='gzip')
            f_out.create_dataset('observation', 
                               shape=(total_transitions, *obs_shape),
                               chunks=(chunk_size, *obs_shape),
                               compression='gzip')
            f_out.create_dataset('next_observation', 
                               shape=(total_transitions, *obs_shape),
                               chunks=(chunk_size, *obs_shape),
                               compression='gzip')
            f_out.create_dataset('qpos', 
                               shape=(total_transitions, *qpos_shape),
                               chunks=(chunk_size, *qpos_shape),
                               compression='gzip')
            f_out.create_dataset('next_qpos', 
                               shape=(total_transitions, *qpos_shape),
                               chunks=(chunk_size, *qpos_shape),
                               compression='gzip')
            f_out.create_dataset('qvel', 
                               shape=(total_transitions, *qvel_shape),
                               chunks=(chunk_size, *qvel_shape),
                               compression='gzip')
            f_out.create_dataset('next_qvel', 
                               shape=(total_transitions, *qvel_shape),
                               chunks=(chunk_size, *qvel_shape),
                               compression='gzip')
            
            # Process each episode and write to output
            current_idx = 0
            for ep_key in episode_keys:
                ep = f_in[ep_key]
                ep_len = len(ep['action'])
                
                # Process this episode in chunks
                for start in range(0, ep_len - 1, chunk_size):
                    end = min(start + chunk_size, ep_len - 1)
                    batch_size = end - start
                    
                    # Get the batch
                    actions = ep['action'][start:end]
                    obs = ep['observation'][start:end]
                    next_obs = ep['observation'][start+1:end+1]
                    qpos = ep['qpos'][start:end]
                    next_qpos = ep['qpos'][start+1:end+1]
                    qvel = ep['qvel'][start:end]
                    next_qvel = ep['qvel'][start+1:end+1]
                    
                    # Write the batch
                    f_out['action'][current_idx:current_idx + batch_size] = actions
                    f_out['observation'][current_idx:current_idx + batch_size] = obs
                    f_out['next_observation'][current_idx:current_idx + batch_size] = next_obs
                    f_out['qpos'][current_idx:current_idx + batch_size] = qpos
                    f_out['next_qpos'][current_idx:current_idx + batch_size] = next_qpos
                    f_out['qvel'][current_idx:current_idx + batch_size] = qvel
                    f_out['next_qvel'][current_idx:current_idx + batch_size] = next_qvel
                    
                    current_idx += batch_size
                    print(f"Processed {current_idx}/{total_transitions} transitions", end='\r')
            
            print(f"\nFinished processing {current_idx} transitions")


def main():
    """
    Process the buffer.hdf5 file efficiently by processing one episode at a time.
    
    Args:
        input_path (str): Path to input buffer.hdf5
        output_path (str): Path to save processed output
        chunk_size (int): Number of transitions to process before writing to disk
    """
    parser = argparse.ArgumentParser(description='Process buffer.hdf5 file efficiently.')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input buffer.hdf5 file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save processed output file')
    parser.add_argument('--chunk-size', type=int, default=10000,
                      help='Number of transitions to process before writing to disk (default: 10000)')
    args =  parser.parse_args()
    print(f"Processing {args.input}...")
    process_buffer_memory_efficient(args.input, args.output, args.chunk_size)
    print(f"Processing complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()