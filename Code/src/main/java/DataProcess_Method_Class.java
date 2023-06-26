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

public class DataProcess_Method_Class {
   
    //全局变量
    private static String sampleClasses;
    private static List<ClassOrInterfaceDeclaration> ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //保存所有外部 class，只要第一个 || 清空、获取/第一个
    private static MethodDeclaration sampleMethod;
    private static String sampleMethodName;
    private static String needRefactmethodName;
    private static boolean exceptionClass = false;
    //原始数据集路径
    private static String rootPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/SourceCode";  //存储待解析Java文件的根目录
    private static String csvPath = "/home/yqx/Documents/my-FeatureEnvy-dataset/groundTruth.csv";
    //数据增强后的数据集路径
    //private static String dataset_Class_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Class_Class";
    private static String dataset_Method_Class_Path = "/home/yqx/Documents/my-FeatureEnvy-dataset/Dataset_Method_Class";
    //数据条目保存路径
    private static String dataItemPath = dataset_Method_Class_Path+'/'+"dataItem"+'/';
    //labels.txt保存路径
    private static String labelTxtPath = dataset_Method_Class_Path+'/'+"labels.txt";
    //统计正负样本数量
    private static int posSamples = 0;
    private static int negSamples = 0;

    private static class MethodNameCollector extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            collector.add(md.getNameAsString());
        }
    }

    private static class getAnMethodByName extends VoidVisitorAdapter<List<String>> { //MethodNameCollector，抽取方法信息
        @Override
        public void visit(MethodDeclaration md, List<String> collector){
            super.visit(md, collector);
            if (md.getNameAsString().equalsIgnoreCase(sampleMethodName)){ // 通过 MethodName 来得到这个方法的代码
                sampleMethod = md; //不能返回值，那就交给全局变量 sampleMethod
                //System.out.println("sampleMethod");
                //System.out.println(sampleMethod);
            }
        }
    }

    private static class ClassNameCollector extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息

        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> methodRemoveList){
            super.visit(n, methodRemoveList);
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
                    ClassList.add(n);
                }
            } catch (Exception e) {
                //TODO: handle exception
                //System.out.println("exception_________exception");
                exceptionClass = true;
            }
        }
    }

    private static class gatAnClassByPath extends VoidVisitorAdapter<List<String>> { //ClassNameCollector，获取Java类信息
        @Override
        public void visit(ClassOrInterfaceDeclaration n, List<String> methodRemoveList){
            super.visit(n, methodRemoveList);
            try {
                if(n.isTopLevelType()){ //保存每个类,但是只要外部类
                    ClassList.add(n); //不能返回值，那就交给全局变量 newClass
                }
            } catch (Exception e) {
                //TODO: handle exception
                //System.out.println("exception_________exception");
                exceptionClass = true;
            }
        }
    }

    public static List<String> getMethodNames(String codePath) throws FileNotFoundException{
        List<String> methodNames = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new DataProcess_Method_Class.MethodNameCollector();
        methodNameVisitor.visit(cp, methodNames);
        return methodNames;
    }

    public static MethodDeclaration getMethodByName(String codePath, String methodName) throws FileNotFoundException{
        sampleMethodName = methodName;
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> methodNameVisitor = new DataProcess_Method_Class.getAnMethodByName();
        methodNameVisitor.visit(cp, new ArrayList<>());
        return sampleMethod;
    }

    public static List<ClassOrInterfaceDeclaration> getNewClassCode(String codePath, List<String> methodRemoveList) throws FileNotFoundException{
        //List<String> newClass = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new DataProcess_Method_Class.ClassNameCollector();
        classNameVisitor.visit(cp, methodRemoveList);
        return ClassList;
    }

    public static List<ClassOrInterfaceDeclaration> getClassByPath(String codePath) throws FileNotFoundException{
        //List<String> newClass = new ArrayList<>();
        CompilationUnit cp = StaticJavaParser.parse(new File(codePath));
        VoidVisitor<List<String>> classNameVisitor = new DataProcess_Method_Class.gatAnClassByPath();
        classNameVisitor.visit(cp, new ArrayList<>());
        return ClassList;
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

    public static void writeSampleMethodToDataItem(String srcJavaSavePath, MethodDeclaration sampleMethod) throws IOException{
        //保存Java，写入本地文件
        BufferedWriter aClass = new BufferedWriter(new FileWriter(srcJavaSavePath,false));
        aClass.write(sampleMethod.toString());
        aClass.newLine();
        aClass.close();
    }
    
    public static void writeSampleClassToDataItem(String classSavePath, String sampleClass) throws IOException{
        BufferedWriter aClass = new BufferedWriter(new FileWriter(classSavePath,false));
        aClass.write(sampleClass.toString());
        aClass.newLine();
        aClass.close();
    }

    public static void appendLabelItem(String labelTxtPath, String labelInfo) throws IOException{
        BufferedWriter txt = new BufferedWriter(new FileWriter(labelTxtPath,true));
        txt.write(labelInfo.toString());
        txt.newLine();
        txt.close();
    }

    public static void saveAnDataItem(String projectName, String srcClassPath, String tagClassPath, String srcClassName, String tagClassName, String needRefactmethodName, String sampleMethodName, String sampleClassName, MethodDeclaration sampleMethod, String sampleClasses, List<String> srcClassMethodNames, List<String> tagClassMethodNames, int item, int label) throws IOException{
        String itemPath;
        if(label == 1)
            itemPath = getDataItemPath(projectName, needRefactmethodName, srcClassName, tagClassName, "pos", item);
        else
            itemPath = getDataItemPath(projectName, needRefactmethodName, srcClassName, tagClassName, "neg", item);
        //正样本中 needRefactmethodName == sampleMethodName ， 反之为负样本 
        String methodSavePath = itemPath + '/' + sampleMethodName + ".java";
        String classSavePath = itemPath + '/' + sampleClassName + ".java";
        String dataJsonInfoSavePath = itemPath + '/' + "dataItemInfo.json";
        //保存 tagClassPathPos 原始代码，写入本地文件
        writeSampleClassToDataItem(classSavePath, sampleClasses);
        //保存 refactMethod 原始代码，写入本地文件
        writeSampleMethodToDataItem(methodSavePath, sampleMethod);
        //生成并保存 dataItemInfo.json 
        JSONObject dataItemJson = new JSONObject();
        dataItemJson.put("projectName",projectName);
        dataItemJson.put("needRefactmethodName",needRefactmethodName);
        dataItemJson.put("sampleMethodName",sampleMethodName);
        dataItemJson.put("srcClassName",srcClassName);
        dataItemJson.put("tagClassName",tagClassName);
        dataItemJson.put("MethodNamesInSrc",srcClassMethodNames);
        dataItemJson.put("MethodNamesInTag",tagClassMethodNames);
        //dataItemJson.put("removedMethodNamesInSrc",removeMethodsInSrc);
        //dataItemJson.put("removedMethodNamesInTag",new ArrayList<String>());
        dataItemJson.put("label",label);
        FileUtils.write(new File(dataJsonInfoSavePath), dataItemJson.toString(), StandardCharsets.UTF_8, true);

        //向 labels.txt 中追加一个 label 条目
        String labelInfo = itemPath.replace(dataset_Method_Class_Path+'/',"")+"  "+sampleMethodName+"  "+sampleClassName+"  "+label; // dataItemPath  SamplemethodName tagClassName 0/1
        appendLabelItem(labelTxtPath, labelInfo);
        if (label == 1){
            posSamples++;
        }else{
            negSamples++;
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
                
                System.out.println(projectName);
                /**
                 * 正样本：
                 * 原始的 needRefactmethodName -> TagClass  1
                 * 增强的 needRefactmethodName -> TagClass-(some 与 needRefactmethodName无关的方法)  1
                 * 增强的 needRefactmethodName -> TagClass-(some 与 needRefactmethodName无关的方法)  1
                 * 
                 * 负样本：
                 *  (each method in classA) & classB  0
                 *  (each method in classB) & classA  0
                 * }
                 */
                
                // 首先考虑正样本，得到了DateItem（a）初步的原始信息
                String srcClassPathPos = rootPath + '/' + projectName + "/a/" + srcClassName + ".java";
                String tagClassPathPos = rootPath  + '/' + projectName + "/a/" + tagClassName + ".java";
                
                sampleMethodName = methodName; // 此时 sampleMethodName 为srcClass中的 refactMethodName ,正样本
                needRefactmethodName = methodName; //正样本中 sampleMethodName == needRefactmethodName
                List<String> srcClassMethodNamesPos = getMethodNames(srcClassPathPos);
                List<String> tagClassMethodNamesPos = getMethodNames(tagClassPathPos);

                
                exceptionClass = false; //重置异常标志--------------------------------------------------------------------
                ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //清空
                sampleClasses = new String();
                ClassList = getClassByPath(tagClassPathPos); //可能引发异常
                int itemPos = 0;
                if (ClassList.size()>0){
                    for (int i=0; i<ClassList.size(); i++){
                        sampleClasses += ClassList.get(i).toString();
                    }
                    sampleMethod = getMethodByName(srcClassPathPos, methodName);

                    //此时，得到了 sampleMethod，只需要从 TagClass 中移除与 sampleMethod 无关的方法来制造新的正样本
                    //首先，保存原始数据条目pos ： 原始的 needRefactmethodName -> TagClass  1
                    //saveAnDataItem(projectName, srcClassPath, tagClassPath, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, sampleClassName, sampleMethod, sampleClass, srcClassMethodNames, tagClassMethodNames, item, label)
                    if(!exceptionClass){
                        saveAnDataItem(projectName, srcClassPathPos, tagClassPathPos, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, tagClassName, sampleMethod, sampleClasses, srcClassMethodNamesPos, tagClassMethodNamesPos, itemPos, 1);
                        itemPos+=1;
                    }
                }
                
                //其次，制造新的增强的正样本：  needRefactmethodName -> TagClass-(some 与 needRefactmethodName无关的方法)  1
                for(int i = 0; i < tagClassMethodNamesPos.size(); i++){
                    if(sampleMethod.toString().contains(tagClassMethodNamesPos.get(i).toString())){
                        tagClassMethodNamesPos.remove(i); //去除 tagClass 中与 sampleMethod 相关的方法, 剩余的删除对 sampleMethod 的影响不大
                    }
                }
                //System.out.println(sampleMethod);
                //System.out.println(tagClassMethodNamesPos);
                List<List<Integer>> removeListsPos = SimpleTesting.getRemoveLists(tagClassMethodNamesPos, 20); //从tagClass中删除一些方法的删除方案
                //System.out.println(removeLists);
                //System.out.println(removeLists.size());
                for (int i = 0; i < removeListsPos.size(); i++){
                    List<String> removeMethodsPos = new ArrayList<String>();
                    for (int index : removeListsPos.get(i)){
                        removeMethodsPos.add(tagClassMethodNamesPos.get(index));
                    }

                    exceptionClass = false; //重置异常标志--------------------------------------------------------------------
                    ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //清空
                    sampleClasses = new String();
                    ClassList = getNewClassCode(tagClassPathPos, removeMethodsPos); //可能引发异常
                    if (ClassList.size()>0){
                        for (int j=0; j<ClassList.size(); j++){
                            sampleClasses += ClassList.get(j).toString();
                        }
                        //System.out.println(removeMethodsPos);
                        if(!exceptionClass){
                            saveAnDataItem(projectName, srcClassPathPos, tagClassPathPos, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, tagClassName, sampleMethod, sampleClasses, srcClassMethodNamesPos, tagClassMethodNamesPos, itemPos, 1);
                            itemPos+=1;
                        }
                    }
                }


                //其次考虑负样本,从重构后的类对中获取，首先考虑两种Case： (each method in classA) & classB || (each method in classB) & classA 
                String srcClassPathNeg = rootPath + '/' + projectName + "/b/" + srcClassName + ".java";
                String tagClassPathNeg = rootPath  + '/' + projectName + "/b/" + tagClassName + ".java";
                List<String> srcClassMethodNamesNeg = getMethodNames(srcClassPathNeg);
                List<String> tagClassMethodNamesNeg = getMethodNames(tagClassPathNeg);
                //System.out.println(srcClassMethodNamesNeg);
                //System.out.println(tagClassMethodNamesNeg);

                int itemNeg = 0;
                //首先是 (each method in classA) & classB  // 是否限定数量？
                //遍历 each method in classA ，与原始 classB 配对组成一条数据
                needRefactmethodName = methodName; //这个不变

                exceptionClass = false; //重置异常标志--------------------------------------------------------------------
                ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //清空
                sampleClasses = new String();
                ClassList = getClassByPath(tagClassPathNeg); //可能引发异常
                if (ClassList.size()>0){
                    for (int j=0; j<ClassList.size(); j++){
                        sampleClasses += ClassList.get(j).toString();
                    }
                    if(!exceptionClass){
                        for (String name : srcClassMethodNamesNeg){
                            sampleMethod = getMethodByName(srcClassPathNeg, name); //each method code in classA
                            sampleMethodName = name;
                            //saveAnDataItem(projectName, srcClassPath, tagClassPath, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, sampleClassName, sampleMethod, sampleClass, srcClassMethodNames, tagClassMethodNames, item, label)
                            saveAnDataItem(projectName, srcClassPathNeg, tagClassPathNeg, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, tagClassName, sampleMethod, sampleClasses, srcClassMethodNamesNeg, tagClassMethodNamesNeg, itemNeg, 0);
                            itemNeg+=1;
                        }
                    }
                }
                //其次是 (each method in classB) & classA   // 是否限定数量？ 如法泡制
                //遍历 each method in classB ，与原始 classA 配对组成一条数据
                needRefactmethodName = methodName; //这个不变

                exceptionClass = false; //重置异常标志--------------------------------------------------------------------
                ClassList = new ArrayList<ClassOrInterfaceDeclaration>(); //清空
                sampleClasses = new String();
                ClassList = getClassByPath(srcClassPathNeg); //可能引发异常
                if (ClassList.size()>0){
                    for (int j=0; j<ClassList.size(); j++){
                        sampleClasses += ClassList.get(j).toString();
                    }
                    if(!exceptionClass){
                        for (String name : tagClassMethodNamesNeg){
                            sampleMethod = getMethodByName(tagClassPathNeg, name); //each method code in classA
                            sampleMethodName = name;
                            saveAnDataItem(projectName, srcClassPathNeg, tagClassPathNeg, srcClassName, tagClassName, needRefactmethodName, sampleMethodName, srcClassName, sampleMethod, sampleClasses, srcClassMethodNamesNeg, tagClassMethodNamesNeg, itemNeg, 0);
                            itemNeg+=1;
                        }
                    }
                }
                //break;
            }
            reader.close();
            System.out.println("正样本数量： "+posSamples); //6005
            System.out.println("负样本数量： "+negSamples); //15023
        } catch (Exception e) {  
            e.printStackTrace(); 
            System.out.println("Some exceptions occure ~"); 
        }  
    }

    public static void main(String[] args) throws FileNotFoundException {
        
        parse_file(csvPath);
    }
}
