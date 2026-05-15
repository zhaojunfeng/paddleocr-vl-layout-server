feature 1: 增加markitdown (https://github.com/microsoft/markitdown)来处理扫描pdf和图片以外其他类型的文件
pdf是否扫描件的判断依据，前两页含有文本字符<50个即判断成扫描件

feature 2:
现在再实现一个批量接口.
1. 用户上传一个压缩包, 自动识别 .zip / .tar.gz / .tar.bz2 / .tar.xz / .tgz
2. 解压缩，解压缩文件夹名是 压缩包文件名+时间戳, 同时在同父目录下创建处理后文件夹名称 压缩包文件名+时间戳+'result', 以备后用
3. 遍历解压文件夹中的所有文件,依次读取或者识别内容。markitdown读取的内容 保存在 reusult目录内，保持和解压后一样的相对目录结构，文件名是：原文件名+'.md'.
   ocr识别的保存两个文件，一个保存文本内容的md文件，另一个保存 layoutParsingResults的json文件，文件名是：原文件名+'.json'.
   读取或识别完一个文件，无论成功失败，都要更新 manifest.json
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
4. 全部读取或识别完成，将result文件打成压缩包 zip.
   删除原始包、extract目录、result目录
   等待用户获取最终压缩包，result压缩包保留7天，过期删除.
   