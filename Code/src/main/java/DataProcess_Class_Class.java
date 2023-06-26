import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.visitor.VoidVisitor;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONObject;

import org.apache.commons.io.FileUtils;
import java.nio.charset.StandardCharsets;

public class DataProcess_Class_Class {
    //全局变量

    private static String sampleClasses;
    private static List<ClassOrInterfaceDeclaration> ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //保存所有外部 class，只要第一个 || 清空、获取/第一个
    //private static ClassOrInterfaceDeclaration newClassAfterMethodRemove;
    private static MethodDeclaration refactMethod;
    //private static String currentClassName;
    private static String refactMethodName;
    private static boolean exceptionClass = false;
    private static String refactMethodNameInPos = "yes"; // 只要src 代码中的 refactMethodName
    //原始数据集路径
    private static String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/SourceCode";  //存储待解析Java文件的根目录
    private static String csvPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/groundTruth.csv";
    //数据增强后的数据集路径
    private static String dataset_Class_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class";
    //private static String dataset_Method_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class";
    //数据条目保存路径
    private static String dataItemPath = dataset_Class_Class_Path+'/'+"dataItem"+'/';
    //labels.txt保存路径
    private static String labelTxtPath = dataset_Class_Class_Path+'/'+"labels.txt";
    //统计正负样本数量
    private static int posSamples = 0;
    private static int negSamples = 0;

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            collector.add(md.getNameAsString());
            //System.out.println(refactMethodName);
            //System.out.println(md.getNameAsString());
            if (md.getNameAsString().equalsIgnoreCase(refactMethodName) && refactMethodNameInPos == "yes"){ // 只要src中的 refactMethod
                refactMethod = md; //不能返回值，那就交给全局变量 refactMethod
                //System.out.println("refactMethod");
                //System.out.println(refactMethod);
            }
        }
    }

    private static class ClassNameCollector extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息

        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> methodRemoveList){
            super.visit(n, methodRemoveList);
            //collector.add(n.getNameAsString());
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
            
            //System.out.println(methodRemoveList);
            //String [] removeMethodNameList = {"setupEnableOtherIfNotZero","addRadioButtons","syncValueToPixels","syncValueToPercentage","componentResized"};
            //System.out.println(tagMethodNameList[0]);
            try {
                for(int i = 0; i < n.getMembers().size(); i++){
                    for(String name: methodRemoveList){
                        //System.out.println(name);
                        for(int j = 0; j < n.getMethodsByName(name).size(); j++){
                            if(n.getMethodsByName(name).get(j) == n.getMembers().get(i)){
                                //System.out.println("remove  "+name);
                                n.getMembers().remove(i);
                            }
                        }
                    }
                }
                if(n.isTopLevelType()){ //保存每个类,但是只要外部类
                    ClassList.add(n);//不能返回值，那就交给全局变量 newClass
                }
                if(n.toString().equalsIgnoreCase("enum"))
                    System.out.println("111111111111111111111111111111111111111111111111111111");

            } catch (Exception e) {
                //TODO: handle exception
                //System.out.println("exception_________exception");
                //出现异常，则跳过此条数据
                exceptionClass = true;
            }
        }
    }

    public static List<String> getMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new DataProcess_Class_Class.MethodNameCollector();
        methodNameVisitor.visit(cp, methodNames);
        return methodNames;
    }

    public static void newClassCode(String codePath, List<String> methodRemoveList) throws FileNotFoundException{
        //List<String> newClass = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new DataProcess_Class_Class.ClassNameCollector();
        classNameVisitor.visit(cp, methodRemoveList);
    }

    public static String getDataItemPath(String projectName, String methodName, String srcClassName, String tagClassName, String posNeg, int itemPos){
        String s;
        if (posNeg == "pos"){
            s = "_item_pos_";
        }else{
            s = "_item_neg_";
        }
        File file = new File(dataItemPath+projectName+'_'+methodName+'_'+srcClassName+'_'+tagClassName +s+ itemPos);
        if (!file.exists()) {
            file.mkdirs();
        }
        return dataItemPath+projectName+'_'+methodName+'_'+srcClassName+'_'+tagClassName +s+ itemPos;
    }

    public static void writeNewClassToDataItem(String classSoursePath, String classSavePath, List<String> removeNameLists) throws IOException{
        sampleClasses = new String();
        ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //清空
        newClassCode(classSoursePath, removeNameLists); //参数2为空，，不删除任何方法
        if (ClassList.size()>0){
            for (int j=0; j<ClassList.size(); j++){
                sampleClasses += ClassList.get(j).toString();
            }
        }else{
            exceptionClass = true;
        }
        if (!exceptionClass){ //无异常时执行
            //保存Java，写入本地文件，有异常时可能错误保存上一个class的代码
            BufferedWriter aClass = new BufferedWriter(new FileWriter(classSavePath,false));
            aClass.write(sampleClasses);
            aClass.newLine();
            aClass.close();
        }
    }

    public static void appendLabelItem(String labelTxtPath, String labelInfo) throws IOException{
        BufferedWriter txt = new BufferedWriter(new FileWriter(labelTxtPath,true));
        txt.write(labelInfo.toString());
        txt.newLine();
        txt.close();
    }

    public static void saveAnDataItem(String projectName, String srcClassPath, String tagClassPath, String methodName, String srcClassName, String tagClassName, List<String> srcClassMethodNames, List<String> tagClassMethodNames, List<String> removeMethodsInSrc, List<String> removeMethodsInTag, int item, int label) throws IOException{
        String itemPath;
        if(label == 1)
            itemPath = getDataItemPath(projectName, methodName, srcClassName, tagClassName, "pos", item);
        else
            itemPath = getDataItemPath(projectName, methodName, srcClassName, tagClassName, "neg", item);
        
        String srcJavaSavePath = itemPath + '/' + srcClassName + ".java";
        String tagJavaSavePath = itemPath + '/' + tagClassName + ".java";
        String dataJsonInfoSavePath = itemPath + '/' + "dataItemInfo.json";

        exceptionClass = false; //重置异常标志
        //保存 srcClassPathPos 原始代码，写入本地文件
        writeNewClassToDataItem(srcClassPath, srcJavaSavePath, removeMethodsInSrc);
        //保存 tagClassPathPos 原始代码，写入本地文件
        writeNewClassToDataItem(tagClassPath, tagJavaSavePath, removeMethodsInTag);
        //生成并保存 dataItemInfo.json 
        /**
         * DateItem：{
         *  projectName: "projectName",
         *  methodName: "methodName",
         *  srcClassName: "srcClassName",
         *  tagClassName: "tagClassName",  
         *  MethodNamesInSrc: [m1,m2,m3,...,m10]，  // 从javaparser获取，来自srcClassPath的方法
         *  MethodNamesInTag: [m1,m2,m3,...,m12]，  // 从javaparser获取，来自tagClassPath的方法
         *  removedMethodNamesInSrc: [m2,m3],
         *  removedMethodNamesInTag: [],
         *  label: 1
         * }
         */
        if (!exceptionClass){ //无异常时保存,有异常则放弃这条数据
            JSONObject dataItemJson = new JSONObject();
            dataItemJson.put("projectName",projectName);
            dataItemJson.put("methodName",methodName);
            dataItemJson.put("srcClassName",srcClassName);
            dataItemJson.put("tagClassName",tagClassName);
            dataItemJson.put("MethodNamesInSrc",srcClassMethodNames);
            dataItemJson.put("MethodNamesInTag",tagClassMethodNames);
            dataItemJson.put("removedMethodNamesInSrc",removeMethodsInSrc);
            dataItemJson.put("removedMethodNamesInTag",removeMethodsInTag);
            dataItemJson.put("label",label);
            FileUtils.write(new File(dataJsonInfoSavePath), dataItemJson.toString(), StandardCharsets.UTF_8, true);

            //向 labels.txt 中追加一个 label 条目
            String labelInfo = itemPath.replace(dataset_Class_Class_Path+'/',"")+"  "+srcClassName+"  "+tagClassName+"  "+label;
            appendLabelItem(labelTxtPath, labelInfo);
            //两个class调换顺序，不一样的输入，一样的结果，同样可以作为一条数据
            String labelInfoR = itemPath.replace(dataset_Class_Class_Path+'/',"")+"  "+tagClassName+"  "+srcClassName+"  "+label;
            appendLabelItem(labelTxtPath, labelInfoR);
            if (label == 1){
                posSamples+=2;
            }else{
                negSamples+=2;
            }
        }else{
            System.out.println(itemPath.replace(dataset_Class_Class_Path+'/',"")+"  "+srcClassName+"  "+tagClassName+"  "+label);
        }
    }

    private static void parse_file(String csvPath) throws FileNotFoundException {

        try {  
            BufferedReader reader = new BufferedReader(new FileReader(csvPath)); 
            reader.readLine();//第一行信息，为标题信息，不用,如果需要，注释掉 
            String line = null;  
            while((line=reader.readLine())!=null){  
                String item[] = line.split(",");//CSV格式文件为逗号分隔符文件
                  
                String projectName = item[1];
                String methodName = item[2];
                String srcClassName = item[3].split("\\/")[item[3].split("\\/").length-1].split("\\.")[0];
                String tagClassName = item[4].split("\\/")[item[4].split("\\/").length-1].split("\\.")[0];

                

                refactMethodName = methodName; // 将目标方法名告诉全局变量，全局变量根据这个名字来获得它的code
                //System.out.println(refactMethodName);
                System.out.println(projectName);

                //System.out.println(methodName);  
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
                
                // 首先考虑正样本，得到了DateItem（a）初步的原始信息
                String srcClassPathPos = rootPath + '/' + projectName + "/a/" + srcClassName + ".java";
                String tagClassPathPos = rootPath  + '/' + projectName + "/a/" + tagClassName + ".java";

                refactMethodNameInPos = "yes";
                List<String> srcClassMethodNamesPos = getMethodNames(srcClassPathPos);
                refactMethodNameInPos = "no";
                List<String> tagClassMethodNamesPos = getMethodNames(tagClassPathPos);
                //从所有方法名中去除 needRefactMethodNameInSrc，以及它调用过的方法，如果移除可能对它有影响,,规定 refactMethod 中没有调用的方法才可以被移除
                for(int i = 0; i < srcClassMethodNamesPos.size(); i++){
                    if(srcClassMethodNamesPos.get(i).equalsIgnoreCase(methodName) || refactMethod.toString().contains(srcClassMethodNamesPos.get(i).toString())){
                        srcClassMethodNamesPos.remove(i); //去除 needRefactMethodNameInSrc, 以及其调用过的方法,如果移除可能对它有影响 
                    }
                }
                //System.out.println(srcClassMethodNamesPos);
                //System.out.println(tagClassMethodNamesPos);

                
                //首先，保存原始数据条目pos
                int itemPos = 0;
                saveAnDataItem(projectName, srcClassPathPos, tagClassPathPos, methodName, srcClassName, tagClassName, srcClassMethodNamesPos, tagClassMethodNamesPos, new ArrayList<String>(), new ArrayList<String>(), itemPos, 1);
                itemPos+=1;
                //然后，根据方法移除方案生成新的数据条目
                List<List<Integer>> removeListsPos = SimpleTesting.getRemoveLists(srcClassMethodNamesPos, 20); // 随机获取20个不重复的移除方案列表
                //System.out.println(removeLists);
                //System.out.println(removeLists.size());
                for (int i = 0; i < removeListsPos.size(); i++){
                    List<String> removeMethodsPos = new ArrayList<String>();
                    for (int index : removeListsPos.get(i)){
                        removeMethodsPos.add(srcClassMethodNamesPos.get(index));
                    }
                    //System.out.println(removeMethodsPos);
                    saveAnDataItem(projectName, srcClassPathPos, tagClassPathPos, methodName, srcClassName, tagClassName, srcClassMethodNamesPos, tagClassMethodNamesPos, removeMethodsPos, new ArrayList<String>(), itemPos, 1);
                    itemPos+=1;
                }

                //其次考虑负样本，得到了DateItem（a）初步的原始信息  ,负样本中规定不能把方法删完了，至少每个类要保留一个
                String srcClassPathNeg = rootPath + '/' + projectName + "/b/" + srcClassName + ".java";
                String tagClassPathNeg = rootPath  + '/' + projectName + "/b/" + tagClassName + ".java";
                //refactMethodNameInPos = "yes";
                List<String> srcClassMethodNamesNeg = getMethodNames(srcClassPathNeg);
                //refactMethodNameInPos = "no";
                List<String> tagClassMethodNamesNeg = getMethodNames(tagClassPathNeg);
                //System.out.println(srcClassMethodNamesNeg);
                //System.out.println(tagClassMethodNamesNeg);
                //首先，保存原始数据条目neg
                int itemNeg = 0;
                saveAnDataItem(projectName, srcClassPathNeg, tagClassPathNeg, methodName, srcClassName, tagClassName, srcClassMethodNamesNeg, tagClassMethodNamesNeg, new ArrayList<String>(), new ArrayList<String>(), itemNeg, 0);
                itemNeg+=1;
                //然后，根据方法移除方案生成新的数据条目
                List<List<Integer>> removeListsNeg = SimpleTesting.getRemoveLists(srcClassMethodNamesNeg, 20); // 1/2 只删除src中的方法 
                //System.out.println(removeLists);
                //System.out.println(removeLists.size());
                for (int i = 0; i < removeListsNeg.size(); i++){
                    List<String> removeMethodsNeg = new ArrayList<String>();
                    for (int index : removeListsNeg.get(i)){
                        removeMethodsNeg.add(srcClassMethodNamesNeg.get(index));
                    }
                    //System.out.println(removeMethodsNeg);
                    saveAnDataItem(projectName, srcClassPathNeg, tagClassPathNeg, methodName, srcClassName, tagClassName, srcClassMethodNamesNeg, tagClassMethodNamesNeg, removeMethodsNeg, new ArrayList<String>(), itemNeg, 0);
                    itemNeg+=1;
                }
                //然后，根据方法移除方案生成新的数据条目
                List<List<Integer>> removeListsNeg2 = SimpleTesting.getRemoveLists(tagClassMethodNamesNeg, 20); // 2/2 只删除tag中的方法 
                for (int i = 0; i < removeListsNeg2.size(); i++){
                    List<String> removeMethodsNeg = new ArrayList<String>();
                    for (int index : removeListsNeg2.get(i)){
                        removeMethodsNeg.add(tagClassMethodNamesNeg.get(index));
                    }
                    //System.out.println(removeMethodsNeg);
                    saveAnDataItem(projectName, srcClassPathNeg, tagClassPathNeg, methodName, srcClassName, tagClassName, srcClassMethodNamesNeg, tagClassMethodNamesNeg, new ArrayList<String>(), removeMethodsNeg, itemNeg, 0);
                    itemNeg+=1;
                }
                //break;
            }
            reader.close();

            System.out.println("正样本数量： "+posSamples); //12778
            System.out.println("负样本数量： "+negSamples); //26192
        } catch (Exception e) {  
            e.printStackTrace(); 
            System.out.println("Some exceptions occure ~"); 
        }  
    }

    public static void main(String[] args) throws FileNotFoundException {
        
        parse_file(csvPath);
    }
}
