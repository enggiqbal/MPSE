//--eval "var g=dept_graph"

x={}
proposals=""
c=1
nodes=""

publication=""

db.currentemplV4.find({'$or':[{dept:/Computer Science/},{dept:/Mathematics/},{dept:/Systems and Industrial Engr/}]},{_id:0, userid:1, shortname:1, dept:1}).forEach(function (obj){ x[obj.userid]={shortname:obj.shortname, pid:c, dept:obj.dept};
nodes=nodes+ "" +c + ' [label="'+obj.shortname+'"]; \n'
c=c+1;
})

print ("Total count:" + c)

var activenodes=[]

db.networkV4.find({userid : {'$in':Object.keys(x)}, usertwoid: {'$in':Object.keys(x)} }).forEach(function(obj){
activenodes.push(obj.userid)
activenodes.push(obj.usertwoid)

if ( obj.source=="proposal"){
	txt="" + x[obj.userid].pid + " -- " + x[obj.usertwoid].pid  + ' [edgetype="type1", color="red"]; \n'
	if (proposals.includes(txt)==false)
	proposals=proposals+txt ;
}
if ( obj.source=="uavitae publication"){
	txt= "" + x[obj.userid].pid + " -- " + x[obj.usertwoid].pid  + ' [edgetype="type2", color="green"]; \n'
	if (publication.includes(txt)==false)
		publication=publication+txt;
}
})

var set = new Set(activenodes);
 activenodes = Array.from(set);

nodes=""
for (var i =0; i< activenodes.length ; i++)
{
//print(x[activenodes[i]].shortname)
nodes=nodes+  x[activenodes[i]].pid + ' [dept="'+x[activenodes[i]].dept +'" , label="'+ x[activenodes[i]].shortname +'"]; \n'
}


function getcluster(x,dept) {
var temp=[];
cedges=""
for (k in x)
if (x[k].dept==dept && activenodes.indexOf(k)>-1 ) temp.push(k)

for (var i=0; i<temp.length; i++)
{
for (var j=i+1; j<temp.length; j++)
{
cedges=cedges+ "" + x[temp[i]].pid + " -- " + x[temp[j]].pid  + ' [edgetype="type3", color="blue"]; \n' ;
}
}
print(dept +":"+ temp.length)
return cedges;
}

clusteredge=getcluster(x,'Computer Science')
clusteredge=clusteredge+getcluster(x,'Mathematics')
clusteredge=clusteredge+getcluster(x,'Systems and Industrial Engr')


totalgraph="graph {" + nodes + proposals + publication + clusteredge+"}"

publication_graph="graph {" + nodes +  publication + "}"
proposals_graph="graph {" + nodes +  proposals + "}"
dept_graph="graph {" + nodes +  clusteredge  + "}"
//print(dept_graph)
print(eval(g))
