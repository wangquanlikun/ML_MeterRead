import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

'''
USAGE: 
python split_dataset.py -i your_data.csv -t 0.25 -r 123 -o1 train_part.csv -o2 test_part.csv
'''

def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV dataset into training and testing sets.")
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input CSV file.')
    parser.add_argument(
        '--test_size', '-t',
        type=float,
        default=0.2,
        help='Proportion of the dataset to include in the test split (float between 0 and 1). Default: 0.2')
    parser.add_argument(
        '--random_state', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility. Default: 42')
    parser.add_argument(
        '--train_output', '-o1',
        type=str,
        default='train.csv',
        help='Filename for the output training CSV. Default: train.csv')
    parser.add_argument(
        '--test_output', '-o2',
        type=str,
        default='test.csv',
        help='Filename for the output testing CSV. Default: test.csv')

    args = parser.parse_args()

    # Read the full dataset
    df = pd.read_csv(args.input)
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns.")

    # Perform the split
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state)

    # Save the splits
    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print(f"Saved training set to '{args.train_output}' with {len(train_df)} samples.")
    print(f"Saved testing set to '{args.test_output}' with {len(test_df)} samples.")


if __name__ == '__main__':
    main()
