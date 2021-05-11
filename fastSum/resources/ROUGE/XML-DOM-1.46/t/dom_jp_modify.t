BEGIN {print "1..16\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use CheckAncestors;
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
<サウスパーク>
<シェフ>
おおっす、みんな
</シェフ>
<子供達>
こんちは シェフ
<ケニー>
ウォワォワー
</ケニー>
<カートマン>
だまれ負け犬
</カートマン>
<カイル>
カートマン、でかけつぅ
</カイル>
</子供達>
</サウスパーク>
END

my $parser = new XML::DOM::Parser;
my $doc = $parser->parse ($str);

my $chef = $doc->getElementsByTagName ("シェフ")->item(0);
my $kenny = $doc->getElementsByTagName ("ケニー")->item(0);
my $children = $doc->getElementsByTagName ("子供達")->item(0);

my $stan = $doc->createElement ("スタン");
$children->appendChild ($stan);
my $snap1 =$doc->toString;

my $stanlist = $doc->getElementsByTagName ("スタン");
assert_ok ($stanlist->getLength == 1);

$children->appendChild ($stan);
$stanlist = $doc->getElementsByTagName ("スタン");
assert_ok ($stanlist->getLength == 1);

my $snap2 = $doc->toString;
assert_ok ($snap1 eq $snap2);

# can't add Attr node directly to Element
my $attr = $doc->createAttribute ("おい", "てめえ");
eval {
    $kenny->appendChild ($attr);
};
assert_ok ($@);

$kenny->appendChild ($stan);
assert_ok ($kenny == $stan->getParentNode);

# force hierarchy exception
eval {
    $stan->appendChild ($kenny);
};
assert_ok ($@);

# force hierarchy exception
eval {
    $stan->appendChild ($stan);
};
assert_ok ($@);

my $frag = $doc->createDocumentFragment;
$frag->appendChild ($stan);
$frag->appendChild ($kenny);
$chef->appendChild ($frag);
assert_ok ($frag->getElementsByTagName ("*")->getLength == 0);
assert_ok (not defined $frag->getParentNode);

my $kenny2 = $chef->removeChild ($kenny);
assert_ok ($kenny == $kenny2);
assert_ok (!defined $kenny->getParentNode);

# force exception - can't have 2 element nodes in a document
eval {
    $doc->appendChild ($kenny);
};
assert_ok ($@);

$doc->getDocumentElement->appendChild ($kenny);
$kenny2 = $doc->getDocumentElement->replaceChild ($stan, $kenny);
assert_ok ($kenny == $kenny2);

$doc->getDocumentElement->appendChild ($kenny);

assert_ok (CheckAncestors::doit ($doc));

$str = $doc->toString;
$str =~ tr/\012/\n/;
$str =~ s/(\&\#(\d+);)/sprintf("%s",charRef2U8($2))/eg;
my $end = <<END;
<サウスパーク>
<シェフ>
おおっす、みんな
</シェフ>
<子供達>
こんちは シェフ

<カートマン>
だまれ負け犬
</カートマン>
<カイル>
カートマン、でかけつぅ
</カイル>
</子供達>
<スタン/><ケニー>
ウォワォワー
</ケニー></サウスパーク>
END

assert_ok ($str eq $end);
