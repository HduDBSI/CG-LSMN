1. groundTruth 保留字段说明，只记录重构事实，即正样本，两阶段的负样本根据正样本产生。

    提交SHA码    项目名称     方法名    源类名     目标类名


2. 文件夹说明
    （一级）项目名称命名的文件夹
        （二级）a: 修改前的java文件（左边一列），正样本
        （二级）b: 将feature envy重构后的java文件（右边一列），负样本

3. 使用方法：
    两阶段分类（好处是减少匹配次数）：
        （1）class-class 代码语义配对 -> 识别可以进行 feature envy 重构的 class pair，
            类似于代码克隆检测；
            ps：正样本：负样本 = 1:1，可设法制造更多负样本
        （2）得到需要重构的 class pairs 后，推荐重构策略，即移动哪个 method 到哪个 class
            具体地，classA.method && classB 匹配，与重构事实相符为正样本，否则为负样本
            ps：正样本 < 负样本，不需要更多负样本。

4. 动机：
    （1）目前还没有根据代码语义对匹配来检测 Feature Envy 的，现有方法（基于度量或文本+度量）准确率还很低，可能存在漏检 （代码片段举例）
    （2）现有特征嫉妒数据集大多是由工具识别收集、或者根据制定规则制作的伪数据集样本，它们的类型可能不具备多样性。因此我们有动力收集一个现实世界真实重构的相关数据集。


5. 检索方法：
    高级检索：https://docs.github.com/en/search-github/searching-on-github/searching-code#search-by-language



feature envy merge:true 1808

feature envy merge:false 2843

feature envy committer-date:>2021-01-01             641  checked

feature envy committer-date:2020-08-06..2021-01-02  782  checked

feature envy committer-date:2020-03-01..2020-08-07  936  checked

feature envy committer-date:2019-01-01..2020-03-02  595  checked

feature envy committer-date:2015-01-01..2019-01-02  925  checked

feature envy committer-date:<2015-01-02             812  checked
         
