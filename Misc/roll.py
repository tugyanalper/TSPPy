import numpy as np

records_array = np.array([1, 2, 3, 1, 1, 3, 4, 3, 2])
idx_sort = np.argsort(records_array)
print idx_sort

sorted_records_array = records_array[idx_sort]
print sorted_records_array

vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                   return_index=True)

print vals
print idx_start
print count
# sets of indices
res = np.split(idx_sort, idx_start[1:])
print res
# filter them with respect to their size, keeping only items occurring more than once

vals = vals[count > 1]
res = filter(lambda x: x.size > 1, res)
print vals
print res