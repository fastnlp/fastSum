BEGIN {print "1..23\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use CheckAncestors;
use CmpDOM;
use utf8;
$loaded = 1;
print "ok 1\n";

my $test = 1;
sub assert_ok
{
    my $ok = shift;
    print "not " unless $ok;
    ++$test;
    print "ok $test\n";
    $ok;
}

sub charRef2U8{
my $charRef = shift;
my $u8;
$charRef = pack("H*",sprintf("%x",$charRef));
  for (my $iLen = 0;$charRef ne "";$charRef = substr($charRef,$iLen)){
    if($charRef =~ /^\x00([\x00-\x7F])/){
      $iLen = 2;
      $u8 .= $1;
    }elsif($charRef =~ /^\x00([\x80-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",
        (ord("\xC0")|
         ((ord($1) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\x01-\x07])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",
        (ord("\xC0")|
         ((ord($1) & 7) << 2) |
         ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\x08-\xD7])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",(ord("\xE0") | ((ord($1) & 240) >> 4)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($1) & 15) << 2) |
        ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\xD8-\xDB])([\x00-\xFF])([\xDC-\xDF])([\x00-\xFF])/){
      $iLen = 4;
      $u8 .= pack("v@",(ord("\xF4") |ord($1) & 3));
      $u8 .= pack("v@",(ord("\x80") |((ord($2) & 252)>> 2)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($2) & 3) << 4) |
        ((ord($3) & 3) << 2) |
        ((ord($4) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80") | (ord($4) & 63)));
    }elsif($charRef =~ /^([\xE0-\xFF])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",(ord("\xE0") | ((ord($1) & 240) >> 4)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($1) & 15) << 2) |
        ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }else{
      die "can\'t convert!\n";
    }
  }
  return $u8;
}

#Test 2

my $str = <<END;
<!DOCTYPE シンプソンズ [
 <!ELEMENT 人物 (#PCDATA)>
 <!ATTLIST 人物
  名前 CDATA #REQUIRED
  髪 (なし|青色|黄色) '黄色'
  性別 CDATA #REQUIRED>
]>
<シンプソンズ>
 <人物 名前="ホーマー" 髪="なし" 性別="男性"/>
 <人物 名前="マージ" 髪="青色" 性別="女性"/>
 <人物 名前="バート" 性別="まだ気にしない"/>
 <人物 名前="リサ" 性別="全然気にしない"/>
</シンプソンズ>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);
assert_ok (not $@);

my $out = $doc->toString;
$out =~ tr/\012/\n/;
$out =~ s/(\&\#(\d+);)/sprintf("%s",charRef2U8($2))/eg;
assert_ok ($out eq $str);

my $root = $doc->getDocumentElement;
my $bart = $root->getElementsByTagName("人物")->item(2);
assert_ok (defined $bart);

my $lisa = $root->getElementsByTagName("人物")->item(3);
assert_ok (defined $lisa);

my $battr = $bart->getAttributes;
assert_ok ($battr->getLength == 3);

my $lattr = $lisa->getAttributes;
assert_ok ($lattr->getLength == 3);

# Use getValues in list context
my @attrList = $lattr->getValues;
assert_ok (@attrList == 3);

my $hair = $battr->getNamedItem ("髪");
assert_ok ($hair->getValue eq "黄色");
assert_ok (not $hair->isSpecified);

my $hair2 = $bart->removeAttributeNode ($hair);
# we're not returning default attribute nodes
assert_ok (not defined $hair2);

# check if hair is still defaulted
$hair2 = $battr->getNamedItem ("髪");
assert_ok ($hair2->getValue eq "黄色");
assert_ok (not $hair2->isSpecified);

# replace default hair with pointy hair
$battr->setNamedItem ($doc->createAttribute ("髪", "つんつん"));
assert_ok ($bart->getAttribute("髪") eq "つんつん");

$hair2 = $battr->getNamedItem ("髪");
assert_ok ($hair2->isSpecified);

# exception - can't share Attr nodes
eval {
    $lisa->setAttributeNode ($hair2);
};
assert_ok ($@);

# add it again - it replaces itself
$bart->setAttributeNode ($hair2);
assert_ok ($battr->getLength == 3);

# (cloned) hair transplant from bart to lisa
$lisa->setAttributeNode ($hair2->cloneNode);
$hair = $lattr->getNamedItem ("髪");
assert_ok ($hair->isSpecified);
assert_ok ($hair->getValue eq "つんつん");

my $doc2 = $doc->cloneNode(1);
my $cmp = new CmpDOM;
unless (assert_ok ($doc->equals ($doc2, $cmp)))
{
    # This shouldn't happen
    print "Context: ", $cmp->context, "\n";
}

assert_ok ($hair->getNodeTypeName eq "ATTRIBUTE_NODE");

$bart->removeAttribute ("髪");

# check if hair is still defaulted
$hair2 = $battr->getNamedItem ("髪");
assert_ok ($hair2->getValue eq "黄色");
assert_ok (not $hair2->isSpecified);
