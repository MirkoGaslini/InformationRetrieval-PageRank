#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Libraries importing:
import findspark
findspark.init()
findspark.find()
import pyspark
findspark.find()
from operator import add
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pandas as pd

#Opening PySpark Session:
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

#Defining computeContribs function:
def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


#Defining parseNeighbors function:
def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = urls.split("	")
    return parts[0], parts[1]

#Defining Diff function between 2 lists:
def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

#Defining intersection function between 2 lists:
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

#Reading all the lines of Google web graph: 
lines = sc.textFile('web-Google.txt')
print("Reading and skipping")
lines = lines.zipWithIndex().filter(lambda tup: tup[1] > 3).map(lambda tup: tup[0])
links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

# Find node count
N = links.count()
print(N)
ranks = links.map(lambda url_tuple: (url_tuple[0], 1.0))

old_ranks = ranks
delta = 1

temp_struct=ranks.map(lambda tupla : tupla[0])
temp_struct_collect=temp_struct.collect()
itPR = 0

#PageRank algorithm.
print("Starting PageRank... ")
while(delta > 1.0e-6):
    print("qui")
    itPR = itPR + 1
    contribs = links.join(old_ranks).flatMap(lambda tupla : computeContribs(tupla[1][0],tupla[1][1]))
    dict_contr=contribs.collectAsMap()
    temp_struct_current=list(dict_contr.keys())
    ris=Diff(temp_struct_collect, temp_struct_current)
    val=intersection(ris, temp_struct_collect)
    result = map(lambda e: (e,0),val) 
    result=list(result)
    result=sc.parallelize(result)
    contribs=contribs.union(result)
    n_ranks = contribs.reduceByKey(add).mapValues(lambda rank: (rank * 0.90) + 0.10)
    if(itPR!=1):
        n_ranks_df = pd.DataFrame(n_ranks.sortByKey().collect(), columns =['Node', 'Score'])
        old_ranks_df = pd.DataFrame(old_ranks.sortByKey().collect(), columns =['Node', 'Score'])
        df1 = abs(n_ranks_df['Score'].sub(old_ranks_df['Score'],axis=0))
        delta=df1.sum()
    old_ranks = n_ranks
    n_ranks=None
    n_ranks_df=None
    old_ranks_df=None
    del contribs
    print("Delta: " , delta)
print("Finish PageRank... ")
print("Number of iterations: ", itPR)


# In[ ]:


#transformation in df to save in CSV
old_ranks_df = pd.DataFrame(old_ranks.sortByKey().collect(), columns =['Node', 'Score'])
old_ranks_df.sort_values(by=['Score'], inplace=True, ascending=False)

# Write CSV 
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame

old_ranks_df.to_csv(path_or_buf="csv_PR_0.10_v3")


# In[ ]:


#Saving PR dataframe
import pickle
with open('dataframePR_0.10_v3.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(old_ranks_df,  f)


# In[ ]:


from pyspark.sql.types import *
from functools import reduce
from pyspark.sql.functions import col, lit, when
from graphframes import *

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return TimestampType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types): 
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)

links = lines.map(lambda urls: parseNeighbors(urls))

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


# In[ ]:


#use this row only if you use the saved resut 
old_ranks_df=old_ranks_df.rename(columns={"Node":"id"})


# In[ ]:


vertices= pandas_to_spark(old_ranks_df)
edges = sqlContext.createDataFrame(links, ["src", "dst"])

#graph graphframes
from graphframes import *
g = GraphFrame(vertices, edges)


# In[ ]:


#analysis connected components
sc.setCheckpointDir("/tmp/graphframes-example-connected-components")
result = g.connectedComponents()

#grouping and counting all the connected components
import pyspark.sql.functions as f
sorted_connected=result.groupBy('component').count().select('component', f.col('count').alias('n')).orderBy('n', ascending=False)


# In[ ]:


#community detection
result2 = g.labelPropagation(maxIter=3)
result2_df = pd.DataFrame(result2.collect(), columns =['id', 'label', 'score'])

#saving result community 
with open('result2_v3.pkl', 'wb') as f:
     pickle.dump(result2_df,  f)
     
#Write CSV 
import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame

result2_df.to_csv(path_or_buf="result2_df_3")


# In[ ]:


#For each label (community) counting how many nodes
import pyspark.sql.functions as f
sorted_l=result2.groupBy('label').count().select('label', f.col('count').alias('n')).orderBy('n', ascending=False)


# In[ ]:


#I check which node belongs to the node to choose the community 
result2.filter(result2.id ==  41909).show()#661424963782


# In[ ]:


#selection of community  
component_selected=result2.filter(result2.label == 661424963782 )
component_selected=component_selected.select("id")

#community's vertices  
result_app=result2.select("id","label")
vertices_sub_graph = vertices.join(result_app, vertices.id ==result_app.id,how="left").drop(result_app.id)
vertices_sub_graph = vertices_sub_graph.filter(vertices_sub_graph.label  ==661424963782) 

#community's edges 
edges_sub_graph = edges.join(result_app, edges.src ==result_app.id,how="left").drop(result_app.id)
edges_sub_graph = edges_sub_graph.withColumnRenamed("label", "labelsrc")
edges_sub_graph = edges_sub_graph.join(result_app, edges_sub_graph.dst == result_app.id,how="left").drop(result_app.id)
edges_sub_graph=edges_sub_graph.filter( (edges_sub_graph.label  == 661424963782) | (edges_sub_graph.label  == 661424963782) )
edges_sub_graph=edges_sub_graph.drop(edges_sub_graph.labelsrc)
edges_sub_graph=edges_sub_graph.drop(edges_sub_graph.label)

def listOfTuples(l1, l2): 
    return list(map(lambda x, y:(x,y), l1, l2)) 

edges_sub_graph_src=list(edges_sub_graph.select('src').toPandas()['src'])
edges_sub_graph_dst=list(edges_sub_graph.select('dst').toPandas()['dst']) 
edges_sub_graph_list_tuple=listOfTuples(edges_sub_graph_src, edges_sub_graph_dst)


# In[ ]:


#list creartioon of vertices for networkX
list_app=[]
for vertex in vertices_sub_graph.collect():
    my_dict = {
      "Score": vertex.Score ,
    }
    my_tupla=(vertex.id,my_dict)
    list_app.append(my_tupla)


# In[ ]:


#list ofvertices 
list_vertices=list_app


# In[ ]:


#creation graph networkX for Cytoscape 
import networkx as nx
G = nx.Graph()
G.add_nodes_from(list_vertices)
G.add_edges_from(edges_sub_graph_list_tuple)

#centrality measures 
betweenness_centrality = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_centrality, "betweenness")
closeness_centrality = nx.closeness_centrality(G)
nx.set_node_attributes(G, closeness_centrality, "closeness")
degreee_centrality = nx.degree_centrality(G)
nx.set_node_attributes(G, degreee_centrality, "degree")


# In[ ]:


G=nx.read_gml("test.gml")
h,a=nx.hits(G, max_iter=100, tol=1e-06, nstart=None, normalized=True)#hubs and authorities


# In[ ]:


nx.set_node_attributes(G, h, name="Hubs")


# In[ ]:


nx.set_node_attributes(G, a, name="Authorities")


# In[ ]:


nx.write_gml(G, "test.gml")


# In[ ]:


#file for Cytoscape 
nx.write_gml(G, "test.gml")

