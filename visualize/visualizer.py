def visualize(G, ranks=None, seed_vector=None):
    if ranks is None:
        ranks = {v: 0 for v in G.nodes()}
    if seed_vector is None:
        seed_vector = {v: 0 for v in G.nodes()}
    
    print('----- Visualizing -----')
    print('Packing data')
    data = {}
    data['nodes'] = [{'id':str(u),'group':ranks[u]+seed_vector[u]} for u in G.nodes()]
    data['links'] = [{'source':str(node1),'target':str(node2),'value':1} for node1,node2 in G.edges()]
    print('Writing to file')
    import json
    with open('visualize/data.json', 'w') as outfile:  
        json.dump(data, outfile)
    print('Running firefox')#Chrome with default settings cannot loading external files from scripts
    import os
    os.system('start firefox.exe "file:///'+os.getcwd()+'/visualize/visualize.html"')


def visualize_clusters(clusters):
    print('----- Visualizing -----')
    print('Packing data')
    data = {}
    data['name'] = ""
    data['children'] = [{"name": "", "children": [{"name": entity, "size": 3} for entity in cluster]} for cluster in clusters]
    import json
    with open('visualize/dataCluster.json', 'w') as outfile:  
        json.dump(data, outfile)
    print('Running firefox')#Chrome with default settings cannot loading external files from scripts
    import os
    os.system('start firefox.exe "file:///'+os.getcwd()+'/visualize/visualizeCluster.html"')
    