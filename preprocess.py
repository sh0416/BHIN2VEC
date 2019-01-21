import os
import pandas as pd

# Load
author_df = pd.read_csv(os.path.join('data', 'dblp', 'author2.txt'), sep='\t', header=None, names=['author'])
paper_df = pd.read_csv(os.path.join('data', 'dblp', 'paper2.txt'), sep='\t', header=None, names=['paper'])
topic_df = pd.read_csv(os.path.join('data', 'dblp', 'topic2.txt'), sep='\t', header=None, names=['topic'])
venue_df = pd.read_csv(os.path.join('data', 'dblp', 'venue2.txt'), sep='\t', header=None, names=['venue'])

# Calculate start idx
author_start_idx = 0
paper_start_idx = len(author_df)
topic_start_idx = len(author_df) + len(paper_df)
venue_start_idx = len(author_df) + len(paper_df) + len(topic_df)

# Create dict
author_dict = {x:i for i, x in enumerate(author_df['author'], start=author_start_idx)}
paper_dict = {x:i for i, x in enumerate(paper_df['paper'], start=paper_start_idx)}
topic_dict = {x:i for i, x in enumerate(topic_df['topic'], start=topic_start_idx)}
venue_dict = {x:i for i, x in enumerate(venue_df['venue'], start=venue_start_idx)}

# Convert to id
author_df['id'] = author_df['author'].apply(author_dict.get)
paper_df['id'] = paper_df['paper'].apply(paper_dict.get)
topic_df['id'] = topic_df['topic'].apply(topic_dict.get)
venue_df['id'] = venue_df['venue'].apply(venue_dict.get)

# Save
author_df.to_csv(os.path.join('data', 'dblp', 'author3.txt'), sep='\t', columns=['id', 'author'], index=False, header=False)
paper_df.to_csv(os.path.join('data', 'dblp', 'paper3.txt'), sep='\t', columns=['id', 'paper'], index=False, header=False)
topic_df.to_csv(os.path.join('data', 'dblp', 'topic3.txt'), sep='\t', columns=['id', 'topic'], index=False, header=False)
venue_df.to_csv(os.path.join('data', 'dblp', 'venue3.txt'), sep='\t', columns=['id', 'venue'], index=False, header=False)

# Load
# Heterogeneous edge
write_df = pd.read_csv(os.path.join('data', 'dblp', 'write2.txt'), sep='\t', header=None, names=['author', 'paper'])
write_df['author'] = write_df['author'].apply(author_dict.get)
write_df['paper'] = write_df['paper'].apply(paper_dict.get)
write_df.to_csv(os.path.join('data', 'dblp', 'write3.txt'), sep='\t', columns=['author', 'paper'], index=False, header=False)

publish_df = pd.read_csv(os.path.join('data', 'dblp', 'publish2.txt'), sep='\t', header=None, names=['venue', 'paper'])
publish_df['venue'] = publish_df['venue'].apply(venue_dict.get)
publish_df['paper'] = publish_df['paper'].apply(paper_dict.get)
publish_df.to_csv(os.path.join('data', 'dblp', 'publish3.txt'), sep='\t', columns=['venue', 'paper'], index=False, header=False)

mention_df = pd.read_csv(os.path.join('data', 'dblp', 'mention2.txt'), sep='\t', header=None, names=['paper', 'topic'])
mention_df['paper'] = mention_df['paper'].apply(paper_dict.get)
mention_df['topic'] = mention_df['topic'].apply(topic_dict.get)
mention_df.to_csv(os.path.join('data', 'dblp', 'mention3.txt'), sep='\t', columns=['topic', 'paper'], index=False, header=False)

# Homogeneous edge
cite_df = pd.read_csv(os.path.join('data', 'dblp', 'cite2.txt'), sep='\t', header=None, names=['paper1', 'paper2'])
cite_df['paper1'] = cite_df['paper1'].apply(paper_dict.get)
cite_df['paper2'] = cite_df['paper2'].apply(paper_dict.get)
cite_df.to_csv(os.path.join('data', 'dblp', 'cite3.txt'), sep='\t', columns=['paper1', 'paper2'], index=False, header=False)
