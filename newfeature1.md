现在的预读取功能是同步的，增加一个批量异步预读取的功能。
预读取功能大概这样的：
1. 把未成功读取过的文件 copy到一个preread+时间戳的文件夹，注意保持以前的相对目录结构，然后打zip压缩包。调用批量接口后zip包可以删除掉。

2. 新增文档读取服务，接口调用，header 要加 Bearer token
2.1 auth-manager-config里加 docReaderServer,下面有host和authToken

2.2 使用3个接口来做异步批量处理 /batch/archive-parsing、GET /batch/archive-status/{job_id}、GET /batch/archive-download/{job_id}
    异步处理要有单独的页面查看状态和结果。即使web关闭了，下次启动后，也会查询未完成任务的状态(archive-status). 这个间隔设置成5分钟即可，不用太频繁.
2.3 接口那边批量完成，就调用下载接口，下载处理后的压缩文件。
    解压后，里面有个manifest.json，结构如下
     ```
   [
	  {
		"original_relative_path": "docs/report.pdf",
		"md_relative_path": "docs/report.md",
		"layout_relative_path": "docs/report.json",
		"status": "success",
		"message":""
	  },
	  {
		"original_relative_path": "receipt/test.png",
		"md_relative_path": "",
		"layout_relative_path": "",
		"status": "failed",
		"message":"ocr with exception time out"
	  },
	]
   ```	
2.4 按cache的逻辑把读取后的文件放入相应的 _ocr-cache文件夹。
    和以前比，多了一个layout_relative_path的json文件，这个文件是为另一个功能服务的-可以在原文件上对应识别的block内容，方便以后一一对比，排查识别错误.
	