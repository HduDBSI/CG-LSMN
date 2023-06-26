import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.List;

/**
 * @class MethodNameCollector，抽取方法名
 */

public class MethodNameCollector extends VoidVisitorAdapter<List<String>> {
    /**
     * 首先，读取并解析原始数据集信息，然后用字典保存，例如 a 中的正样本：
     * DateItem（a）：{
     *  projectName: "projectName",
     *  needRefactMethodNameInSrc: "needRefactMethodNameInSrc",
     *  srcClassPath: "srcClassPath",
     *  tagClassPath: "tagClassPath",  //以上是已知的，从excel读取
     *  otherMethodNamesInSrc: [m1,m2,m3,...,m10]，  // 从javaparser获取，来自srcClassPath的方法
     *  MethodNamesInTag: [m1,m2,m3,...,m12]，  // 从javaparser获取，来自tagClassPath的方法
     *  label: 1
     * }
     * 
     * 和 b 中的负样本
     * DateItem（b）：{
     *  projectName: "projectName",
     *  refactedMethodNameInSrc: "refactedMethodNameInSrc",
     *  srcClassPath: "srcClassPath",
     *  tagClassPath: "tagClassPath",  //以上是已知的，从excel读取
     *  MethodNamesInSrc: [m1,m2,m3,...,m10]，  // 从javaparser获取，来自srcClassPath的方法
     *  MethodNamesInTag: [m1,m2,m3,...,m12]，  // 从javaparser获取，来自tagClassPath的方法
     *  label: 0
     * }
     * 
     * 然后，通过移除正负样本 类代码中的无关方法来构造新的正负样本用于类间匹配
     * 
     * 最后，对于重构机会推荐模型，只需通过从正样本中采样的方法来制造正负样本，即
     * needRefactMethodInSrc <---> tagClass 为一个正样本
     * otherMethodInSrc <---> tagClass 为一个负样本
     * 但是，存在的问题是，这样的样本没经过增强，正样本数量很少，那么是否可以从 tagClass 随机移除一些与 needRefactMethodInSrc 无关的方法来增加正样本数量？
     * 例如：tagClass中一些方法的 name 并没有出现在 needRefactMethodInSrc 中，那么这些方法大概率与 needRefactMethodInSrc 是否需要重构是无关的，删除之
     * 后同样达到数据增强的目的，yes，it is！！
     *  
     * 因此，本部分代码只需要提取以上字典中原始正负样本的原始信息，保存在json文件中，进一步的数据增强处理在 ClassNameCollector 中实现，以生成增强后的数据集
     */

    @Override
    public void visit(MethodDeclaration md, List<String> collector){
        super.visit(md, collector);
        collector.add(md.getNameAsString());
        System.out.println(md.getNameAsString()); //打印出每个方法
    }

}
