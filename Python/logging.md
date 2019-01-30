
# 全局配置(basicConfig)

- 用法示例：
    ```python
    import logging

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info('This is a log info')
    logger.debug('Debugging')
    logger.warning('Warning exists')
    logger.info('Finish')
    ```
    在这里我们首先引入了 `logging` 模块，然后进行了一下基本的配置，这里通过 `basicConfig` 配置了 `level` 信息和 `format` 信息，这里 `level` 配置为 INFO 信息，即只输出 INFO 级别的信息，另外这里指定了 `format` 格式的字符串，包括 `asctime、name、levelname、message` 四个内容，分别代表运行时间、模块名称、日志级别、日志内容，这样输出内容便是这四者组合而成的内容了，这就是 `logging` 的全局配置。

    接下来声明了一个 `Logger` 对象，它就是日志输出的主类，调用对象的 `info()` 方法就可以输出 INFO 级别的日志信息，调用 `debug()` 方法就可以输出 DEBUG 级别的日志信息，非常方便。在初始化的时候我们传入了模块的名称，这里直接使用 `__name__` 来代替了，就是模块的名称，如果直接运行这个脚本的话就是 `__main__`，如果是 import 的模块的话就是被引入模块的名称，这个变量在不同的模块中的名字是不同的，所以一般使用 `__name__` 来表示就好了，再接下来输出了四条日志信息，其中有两条 INFO、一条 WARNING、一条 DEBUG 信息，我们看下输出结果：
    ```log
    2019-01-25 09:54:43,404 - __main__ - INFO - This is a log info
    2019-01-25 09:54:43,409 - __main__ - WARNING - Warning exists
    2019-01-25 09:54:43,409 - __main__ - INFO - Finish
    ```
    可以看到输出结果一共有三条日志信息，每条日志都是对应了指定的格式化内容，另外我们发现 DEBUG 的信息是没有输出的，这是因为我们在全局配置的时候设置了输出为 INFO 级别，所以 DEBUG 级别的信息就被过滤掉了。

- 参数设置：

  - filename：即日志输出的文件名，如果指定了这个信息之后，实际上会启用 FileHandler，而不再是 StreamHandler，这样日志信息便会输出到文件中了。**如果不指定，那么就会在控制台打印日志**
  - filemode：这个是指定日志文件的写入方式，有两种形式，一种是 `w`，一种是 `a`，分别代表清除后写入和追加写入。
  - format：指定日志信息的输出格式，即上文示例所示的参数，详细参数可以参考：[python library](https://docs.python.org/3/library/logging.html?highlight=logging%20threadname#logrecord-attributes)，部分参数如下所示：
    - %(levelno)s：打印日志级别的数值。
    - %(levelname)s：打印日志级别的名称。
    - %(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]。
    - %(filename)s：打印当前执行程序名。
    - %(funcName)s：打印日志的当前函数。
    - %(lineno)d：打印日志的当前行号。
    - %(asctime)s：打印日志的时间。
    - %(thread)d：打印线程ID。
    - %(threadName)s：打印线程名称。
    - %(process)d：打印进程ID。
    - %(processName)s：打印线程名称。
    - %(module)s：打印模块名称。
    - %(message)s：打印日志信息。
  - datefmt：指定时间的输出格式，如`datefmt='%Y/%m/%d %H:%M:%S'`。
  - style：如果 format 参数指定了，这个参数就可以指定格式化时的占位符风格，如 `%、{、$` 等。
  - level：指定日志输出的类别，程序会输出大于等于此级别的信息。
  - stream：在没有指定 filename 的时候会默认使用 StreamHandler，这时 stream 可以指定初始化的文件流。
  - handlers：可以指定日志处理时所使用的 Handlers，必须是可迭代的。

# 灵活配置

## Level

logging 模块提供了以下日志等级，每个等级对应一个值：
|     等级     |  值   |
| :----------: | :---: |
| CRITAL/FATAL |  50   |
|    ERROR     |  40   |
| WANRING/WARN |  30   |
|     INFO     |  20   |
|    DEBUG     |  10   |
|    NOTSET    |   0   |
<font color =red>设置 level 后，只有大于等于当前等级的日志才会输出。</font>

## Handle

- 使用示例
    ```python
    import logging
 
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('output.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info('This is a log info')
    logger.debug('Debugging')
    logger.warning('Warning exists')
    logger.info('Finish')
    ```
    这里我们没有再使用 basicConfig 全局配置，而是先声明了一个 Logger 对象，然后指定了其对应的 Handler 为 FileHandler 对象，然后 Handler 对象还单独指定了 Formatter 对象单独配置输出格式，最后给 Logger 对象添加对应的 Handler 即可，最后可以发现日志就会被输出到 output.log 中，内容如下：
    ```log
    2019-01-25 10:36:17,305 - __main__ - INFO - This is a log info
    2019-01-25 10:36:17,305 - __main__ - WARNING - Warning exists
    2019-01-25 10:36:17,305 - __main__ - INFO - Finish
    ```
- Handle
    - StreamHandler：`logging.StreamHandler`；日志输出到流，可以是 `sys.stderr`，`sys.stdout` 或者文件。
    - FileHandler：`logging.FileHandler`；日志输出到文件。
    - BaseRotatingHandler：`logging.handlers.BaseRotatingHandler`；基本的日志回滚方式。
    - RotatingHandler：`logging.handlers.RotatingHandler`；日志回滚方式，支持日志文件最大数量和日志文件回滚。
    - TimeRotatingHandler：`logging.handlers.TimeRotatingHandler`；日志回滚方式，在一定时间区域内回滚日志文件。
    - SocketHandler：`logging.handlers.SocketHandler`；远程输出日志到TCP/IP sockets。
    - DatagramHandler：`logging.handlers.DatagramHandler`；远程输出日志到UDP sockets。
    - SMTPHandler：`logging.handlers.SMTPHandler`；远程输出日志到邮件地址。
    - SysLogHandler：`logging.handlers.SysLogHandler`；日志输出到syslog。
    - NTEventLogHandler：`logging.handlers.NTEventLogHandler`；远程输出日志到Windows NT/2000/XP的事件日志。
    - MemoryHandler：`logging.handlers.MemoryHandler`；日志输出到内存中的指定buffer。
    - HTTPHandler：`logging.handlers.HTTPHandler`；通过”GET”或者”POST”远程输出到HTTP服务器。

    使用三个 Handler 来实现日志同时输出到控制台、文件、HTTP 服务器：
    ```python
    import logging
    from logging.handlers import HTTPHandler
    import sys
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)
    
    # FileHandler
    file_handler = logging.FileHandler('output.log')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # HTTPHandler
    http_handler = HTTPHandler(host='localhost:8001', url='log', method='POST')
    logger.addHandler(http_handler)
    
    # Log
    logger.info('This is a log info')
    logger.debug('Debugging')
    logger.warning('Warning exists')
    logger.info('Finish')
    ```
    运行之前我们需要先启动 HTTP Server，并运行在 8001 端口，其中 log 接口是用来接收日志的接口。

    运行之后控制台输出会输出如下内容：
    ```log
    This is a log info
    Debugging
    Warning exists
    Finish
    ```
    output.log 文件会写入如下内容：
    ```log
    2019-01-25 10:37:08,107 - __main__ - INFO - This is a log info
    2019-01-25 10:37:08,107 - __main__ - WARNING - Warning exists
    2019-01-25 10:37:08,108 - __main__ - INFO - Finish
    ```
    控制台会收到和log文件中一样的信息

    注意的是，在这里 StreamHandler 对象我们没有设置 Formatter，因此控制台只输出了日志的内容，而没有包含时间、模块等信息，而 FileHandler 我们通过 setFormatter() 方法设置了一个 Formatter 对象，因此输出的内容便是格式化后的日志信息。另外每个 Handler 还可以设置 level 信息，最终输出结果的 level 信息会取 Logger 对象的 level 和 Handler 对象的 level 的交集。

## Formatter

在进行日志格式化输出的时候，我们可以不借助于 basicConfig 来全局配置格式化输出内容，可以借助于 Formatter 来完成， Formatter 的用法：
```python
import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                              datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
 
# Log
logger.debug('Debugging')
logger.critical('Critical Something')
logger.error('Error Occurred')
logger.warning('Warning exists')
logger.info('Finished')
```
在这里我们指定了一个 Formatter，并传入了 fmt 和 datefmt 参数，这样就指定了日志结果的输出格式和时间格式，然后 handler 通过 `setFormatter()` 方法设置此 Formatter 对象即可，输出结果如下：
```log
2019/01/25 10:38:28 - __main__ - CRITICAL - Critical Something
2019/01/25 10:38:28 - __main__ - ERROR - Error Occurred
2019/01/25 10:38:28 - __main__ - WARNING - Warning exists
```
这样我们可以每个 Handler 单独配置输出的格式，非常灵活。

## 捕获Traceback

如果遇到错误，我们更希望报错时出现的详细 Traceback 信息，便于调试，利用 logging 模块我们可以非常方便地实现这个记录：
```python
import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
 
# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
# FileHandler
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
 
# StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
 
# Log
logger.info('Start')
logger.warning('Something maybe fail.')
try:
    result = 10 / 0
except Exception:
    logger.error('Faild to get result', exc_info=True)
logger.info('Finished')
```
这里我们在 `error()` 方法中添加了一个参数，将 `exc_info` 设置为了 True，这样我们就可以输出执行过程中的信息了，即完整的 Traceback 信息。
```log
2019-01-25 10:39:06,812 - __main__ - INFO - Start
2019-01-25 10:39:06,812 - __main__ - WARNING - Something maybe fail.
2019-01-25 10:39:06,814 - __main__ - ERROR - Faild to get result
Traceback (most recent call last):
  File "d:/Notes/Notes/t.py", line 24, in <module>
    result = 10 / 0
ZeroDivisionError: division by zero
2019-01-25 10:39:06,815 - __main__ - INFO - Finished
```
## 共享配置

在写项目的时候，我们肯定会将许多配置放置在许多模块下面，这时如果我们每个文件都来配置 logging 配置那就太繁琐了，logging 模块提供了父子模块共享配置的机制，会根据 Logger 的名称来自动加载父模块的配置。
```python
'main.py'

import logging
import core
 
logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
 
# Handler
handler = logging.FileHandler('result.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
 
logger.info('Main Info')
logger.debug('Main Debug')
logger.error('Main Error')
core.run()
```
这里我们配置了日志的输出格式和文件路径，同时定义了 Logger 的名称为 main，然后引入了另外一个模块 core，最后调用了 core 的 run() 方法。
```python
'core.py'

import logging
 
logger = logging.getLogger('main.core')
 
def run():
    logger.info('Core Info')
    logger.debug('Core Debug')
    logger.error('Core Error')
```
这里我们定义了 Logger 的名称为 main.core，注意这里开头是 main，即刚才我们在 main.py 里面的 Logger 的名称，这样 core.py 里面的 Logger 就会复用 main.py 里面的 Logger 配置，而不用再去配置一次了。运行之后会生成一个 result.log 文件，内容如下：
```log
2019-01-25 10:40:53,363 - main - INFO - Main Info
2019-01-25 10:40:53,363 - main - ERROR - Main Error
2019-01-25 10:40:53,364 - main.core - INFO - Core Info
2019-01-25 10:40:53,364 - main.core - ERROR - Core Error
```

## 文件配置

在开发过程中，将配置在代码里面写死并不是一个好的习惯，更好的做法是将配置写在配置文件里面，我们可以将配置写入到配置文件，然后运行时读取配置文件里面的配置，这样是更方便管理和维护的，下面我们以一个实例来说明一下，首先我们定义一个 yaml 配置文件：
```yaml
version: 1
formatters:
  brief:
    format: "%(asctime)s - %(message)s"
  simple:
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
handlers:
  console:
    class : logging.StreamHandler
    formatter: brief
    level   : INFO
    stream  : ext://sys.stdout
  file:
    class : logging.FileHandler
    formatter: simple
    level: DEBUG
    filename: debug.log
  error:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: simple
    filename: error.log
    maxBytes: 10485760
    backupCount: 20
    encoding: utf8
loggers:
  main.core:
    level: DEBUG
    handlers: [console, file, error]
root:
  level: DEBUG
  handlers: [console]
```

这里我们定义了 formatters、handlers、loggers、root 等模块，实际上对应的就是各个 Formatter、Handler、Logger 的配置，参数和它们的构造方法都是相同的。接下来我们定义一个主入口文件，main.py，内容如下：
```python
'main.py'

import logging
import core
import yaml
import logging.config
import os
 
 
def setup_logging(default_path='config.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
 
 
def log():
    logging.debug('Start')
    logging.info('Exec')
    logging.info('Finished')
 
 
if __name__ == '__main__':
    yaml_path = 'config.yaml'
    setup_logging(yaml_path)
    log()
    core.run()
```

这里我们定义了一个 `setup_logging()` 方法，里面读取了 yaml 文件的配置，然后通过 `dictConfig()` 方法将配置项传给了 logging 模块进行全局初始化。另外这个模块还引入了另外一个模块 core，所以我们定义 core.py 如下：
```python
'core.py'

import logging
 
logger = logging.getLogger('main.core')
 
def run():
    logger.info('Core Info')
    logger.debug('Core Debug')
    logger.error('Core Error')
```

观察配置文件，主入口文件 main.py 实际上对应的是 root 一项配置，它指定了 handlers 是 console，即只输出到控制台。另外在 loggers 一项配置里面，我们定义了 main.core 模块，handlers 是 console、file、error 三项，即输出到控制台、输出到普通文件和回滚文件。这样运行之后，我们便可以看到所有的运行结果输出到了控制台：
```log
2019-01-25 11:00:52,062 - Exec
2019-01-25 11:00:52,062 - Finished
2019-01-25 11:00:52,062 - Core Info
2019-01-25 11:00:52,062 - Core Info
2019-01-25 11:00:52,063 - Core Error
2019-01-25 11:00:52,063 - Core Error
```
在 debug.log 文件中则包含了 core.py 的运行结果：
```log
2019-01-25 11:00:52,062 - main.core - INFO - Core Info
2019-01-25 11:00:52,062 - main.core - DEBUG - Core Debug
2019-01-25 11:00:52,063 - main.core - ERROR - Core Error
```