#include "et_def.pb.h"
#include <google/protobuf/stubs/common.h>
#include <iostream>
#include <string>
int main(){
 GOOGLE_PROTOBUF_VERIFY_VERSION;
 ChakraProtoMsg::Node original;original.set_id(42);original.set_name("current-joint-schema-smoke");original.add_data_deps(7);
 std::string encoded;if(!original.SerializeToString(&encoded))return 1;
 ChakraProtoMsg::Node restored;if(!restored.ParseFromString(encoded))return 2;
 if(restored.id()!=42 || restored.name()!=original.name() || restored.data_deps_size()!=1 || restored.data_deps(0)!=7)return 3;
 std::cout<<"protobuf C++ schema roundtrip passed; header_version="<<GOOGLE_PROTOBUF_VERSION<<" bytes="<<encoded.size()<<'\n';
}
