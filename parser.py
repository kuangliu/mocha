# ==========================================================================
# Parse .prototxt file to get layer configuration parameters.
# return a dict containing:
#       { layer_name: { param1:value1, param2:value2, ... } }
# --------------------------------------------------------------------------
# I'm writing the parser on my own, instead of using Google protobuf API
# Cause... it's really hard to get it work with Caffe prototxt. And simple
# text parser should be enough to get the job done.
# ==========================================================================
import re


class PrototxtParser:
    def __init__(self, prototxt):
        print('==> load prototxt..')
        f = open(prototxt, 'r')
        # read file content to a single string
        c = f.read()
        # remove dulplicate spaces
        cc = re.sub('%s+', ' ', c).strip()
        # split by layer
        print('==> split layers..')
        sp = cc.split('layer {')
        sp = [x.strip() for x in sp]
        # parse lines
        self.net = {}
        for i,line in enumerate(sp):
            print('==> parsing line ' + str(i) + '..')
            layer = self.__parse_line(line)
            if 'name' in layer:   # every layer has a name
                layer_name = layer['name']
                print('find layer: '+layer_name)
                self.net[layer_name] = {}
                for k,v in layer.items():
                    self.net[layer_name][k] = v

    def get_param(self, layer_name, param_type, param_name, optional_name=None):
        '''Return param value of [layer_name].[param_type].[param_name]
                              or [layer_name].[param_type].[optional_name]
        e.g. conv1.convolution_param.pad_h
        '''
        assert layer_name in self.net
        if param_name in self.net[layer_name][param_type]:
            return self.net[layer_name][param_type][param_name]
        elif optional_name in self.net[layer_name][param_type]:
            return self.net[layer_name][param_type][optional_name]
        else:
            print('[ERROR] No param '+param_name+' or '+optional_name+' in layer '+layer_name)
            assert False

    def __parse_line(self, line):
        ''' Recursive function converting prototxt layer string to dict.
        e.g. convert 'a:1\n b:2\n c {\n d:1\n e:2\n}'
        to {'a': 1, 'b':2, 'c': {'d':1, 'e': 2}}
        '''
        # remove '}'
        # '}' is useless in parsing and will effect string split
        line = line.replace('}','')

        ret = {}
        # check if the line has nested structure, like {..{..}..}
        idx = line.find('{')
        if idx == -1:  # not nested, this is the base case
            sp = line.strip().split('\n')
            sp = [x.strip() for x in sp if ':' in x]
            for item in sp:
                key, value = self.__split_item(item)
                ret[key] = value
        else:  # nested
            first = line[:idx].strip()
            last = line[idx+1:].strip()
            ret = self.__parse_line(first)
            last_name = first[first.rfind('\n'):idx].strip()
            ret[last_name] = self.__parse_line(last)   # recursive call
        return ret

    def __split_item(self, s):
        """Split item str to dict.
        e.g. ''a':1' -> {'a':1}
        """
        sp = s.split(':')
        sp = [x.strip('\"\'\ ') for x in sp]   # remove auxiliary characters
        assert len(sp) == 2 and sp[0] != '' and sp[1] != ''
        return sp[0].strip(), self.__cast(sp[1])

    def __cast(self, s):
        '''Cast str to str/int/float
        e.g. 'abc'  ->  'abc'   (str)
             '123'  ->   123    (int)
             '12.3' ->  12.3  (float)
        '''
        if all(c in "0123456789.+-" for c in s) and any(c in "0123456789" for c in s):
            if '.' in s:
                s = float(s)
            else:
                s = int(s)
        return s


# p = PrototxtParser('./model/net.t7.prototxt')
