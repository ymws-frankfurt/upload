Batch number: 0
Traceback (most recent call last):
  File "/app/src/pc3_src/step1_tick2bar_binance_ver3.5.py", line 223, in <module>
    bbb.build_time_bars(
  File "/app/src/pc3_src/step1_tick2bar_binance_ver3.5.py", line 93, in build_time_bars
    self.build_bars(get_time_bars_appending, output_path, data_source, **kwargs)
  File "/app/src/pc3_src/step1_tick2bar_binance_ver3.5.py", line 77, in build_bars
    decorated_bar_func(
  File "/app/src/pc3_src/step1_tick2bar_binance_ver3.5.py", line 193, in wrapper
    return func(df, *args, **kwargs)
  File "/app/scripts/jmrichardson_mlfinlab/mlfinlab/data_structures/bar_generators.py", line 131, in get_time_bars_appending
    return bars.batch_run(file_path_or_df=file_path_or_df,
  File "/app/scripts/jmrichardson_mlfinlab/mlfinlab/data_structures/bar_generators.py", line 105, in batch_run
    list_bars = self.run(data=batch)
  File "/app/scripts/jmrichardson_mlfinlab/mlfinlab/data_structures/base_bars.py", line 171, in run
    list_bars = self._extract_bars(data=values)
  File "/app/scripts/jmrichardson_mlfinlab/mlfinlab/data_structures/time_data_structures.py", line 64, in _extract_bars
    date_time = row[0].timestamp()  # Convert to UTC timestamp
AttributeError: 'numpy.float64' object has no attribute 'timestamp'
