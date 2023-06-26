import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import java.util.List;

/**
 * @class ClassNameCollector，获取Java类名
 */

public class ClassNameCollector extends VoidVisitorAdapter<List<String>> {

    @Override
    public void visit(ClassOrInterfaceDeclaration n, List<String> collector){
        super.visit(n, collector);
        collector.add(n.getNameAsString());

        
        //从类中删除一系列想要删除的方法,也可以是内部类的一些方法，并保存删除那些方法后的外部类
        /**
         * 首先，从MethodNameCollector中构造DateItem，用字典保存，例如：
         * DateItem（i）：{
         *  projectName: "projectName",
         *  refactMethodName: "refactMethodName",
         *  srcClassPath: "srcClassPath",
         *  tagClassPath: "tagClassPath",  //以上是已知的，从excel读取
         *  otherMethodNames: [m1,m2,m3,...,m10]，  // 从javaparser获取，来自srcClassPath的方法
         *  //removedMethodNames: [m3,m4,m7]  //(记录)随机删除otherMethodNames中的方法，用于（数据处理阶段直接）产生新样本
         * }
         * 
         * 然后，根据 DateItem 中的 removedMethodNames 列表来随机从外部类代码中删除它们，保存增强后的外部类，和tagClass一起构成与DateItem一一对应的新的正样本，保存
         * 
         * 最后，根据正样本数量，构造数量足够的负样本
         */
        
        String [] removeMethodNameList = {"setupEnableOtherIfNotZero","addRadioButtons","syncValueToPixels","syncValueToPercentage","componentResized"};
        //System.out.println(tagMethodNameList[0]);
        for(int i = 0; i < n.getMembers().size(); i++){
            for(int k = 0; k < removeMethodNameList.length; k++){
                for(int j = 0; j < n.getMethodsByName(removeMethodNameList[k]).size(); j++){
                    if(n.getMethodsByName(removeMethodNameList[k]).get(j) == n.getMembers().get(i)){
                        System.out.println("1111111111111111111111111111111111111111111");
                        n.getMembers().remove(i);
                    }
                }
            }
        }
        System.out.println("1111111111111111111111111111111111111111111");
        if(n.isTopLevelType()){ //保存每个类,但是只要外部类
            System.out.println(n.getNameAsString()); 
        }
    }


    //打印元素类型名
    //private static String getType(Object a) {
    //    return a.getClass().toString();
    //}

}
