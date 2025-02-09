import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
import os

class TrainingVisualizer:
    def __init__(self, output_dir='results'):
        """
        Initialize training visualizer
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize metrics storage
        self.metrics = {
            'cluster_sizes': [],
            'similarity_stats': [],
            'processing_time': [],
            'memory_usage': []
        }
    
    def plot_cluster_distribution(self, labels, save=True):
        """Plot DBSCAN cluster distribution"""
        plt.figure(figsize=(12, 6))
        
        # Count cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = pd.Series(counts, index=unique_labels)
        
        # Store metrics
        self.metrics['cluster_sizes'] = cluster_sizes.to_dict()
        
        # Create bar plot
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title('Distribution of Cluster Sizes')
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Products')
        
        if save:
            plt.savefig(f'{self.output_dir}/cluster_distribution_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()
    
    def plot_similarity_heatmap(self, similarity_matrix, sample_size=100, save=True):
        """
        Plot similarity matrix heatmap
        
        Args:
            similarity_matrix: numpy array of similarity values
            sample_size: size of sample to plot
            save: whether to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Store metrics
        self.metrics['similarity_stats'] = {
            'min': float(np.min(similarity_matrix)),
            'max': float(np.max(similarity_matrix)),
            'mean': float(np.mean(similarity_matrix)),
            'std': float(np.std(similarity_matrix))
        }
        
        # Create heatmap
        sns.heatmap(similarity_matrix, cmap='YlOrRd')
        plt.title(f'Similarity Matrix Heatmap (Sample of {sample_size} items)')
        
        if save:
            plt.savefig(f'{self.output_dir}/similarity_heatmap_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()
    
    def log_processing_time(self, stage_name, time_taken):
        """Log processing time for different stages"""
        self.metrics['processing_time'].append({
            'stage': stage_name,
            'time': time_taken
        })
    
    def log_memory_usage(self, stage_name, memory_mb):
        """Log memory usage for different stages"""
        self.metrics['memory_usage'].append({
            'stage': stage_name,
            'memory_mb': memory_mb
        })
    
    def plot_processing_time(self, save=True):
        """Plot processing time for different stages"""
        if not self.metrics['processing_time']:
            return
            
        df = pd.DataFrame(self.metrics['processing_time'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='stage', y='time', data=df)
        plt.title('Processing Time by Stage')
        plt.xlabel('Processing Stage')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        if save:
            plt.savefig(f'{self.output_dir}/processing_time_{self.timestamp}.png')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self):
        """Generate and save training report"""
        report = pd.DataFrame([
            ["Number of Clusters", len(self.metrics['cluster_sizes'])],
            ["Largest Cluster Size", max(self.metrics['cluster_sizes'].values())],
            ["Average Similarity", self.metrics['similarity_stats']['mean']],
            ["Total Processing Time", sum(m['time'] for m in self.metrics['processing_time'])],
            ["Peak Memory Usage (MB)", max(m['memory_mb'] for m in self.metrics['memory_usage'])]
        ], columns=['Metric', 'Value'])
        
        # Save report
        report.to_csv(f'{self.output_dir}/training_report_{self.timestamp}.csv', index=False)
        
        # Print report
        print("\nTraining Report:")
        print("="*50)
        print(report.to_string(index=False))
        print("="*50)
        
        return report 